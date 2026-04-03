// 复利引擎启动器 - C# Win32 GUI (零闪动) v11
// 编译: C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe /target:winexe /win32icon:icon.ico Launcher.cs
//
// v11：在 v10 基础上加浏览器去重标志文件，彻底解决疯狂双击开多个标签的问题。
// 无论端口检测还是 Mutex 等待，打开浏览器前先检查 .browser_opened 文件，
// 10秒内已开过就跳过。C# 层和 Python 层共用同一个标志文件。

using System;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;

class Program
{
    const string MUTEX_NAME = "Global\\CompoundEngineMutex";
    const int PORT = 7788;
    static readonly string URL = "http://localhost:" + PORT.ToString();
    const string BROWSER_FLAG = ".browser_opened";
    const int BROWSER_COOLDOWN_MS = 10000; // 10秒内不重复开浏览器

    [DllImport("kernel32.dll")]
    static extern IntPtr CreateMutexW(IntPtr lpMutexAttributes, bool bInitialOwner, string lpName);

    [DllImport("kernel32.dll")]
    static extern bool ReleaseMutex(IntPtr hMutex);

    [DllImport("kernel32.dll")]
    static extern bool CloseHandle(IntPtr hObject);

    [DllImport("kernel32.dll")]
    static extern uint GetLastError();

    const uint ERROR_ALREADY_EXISTS = 183;

    static bool IsServerRunning()
    {
        try
        {
            using (var client = new TcpClient())
            {
                var result = client.BeginConnect("127.0.0.1", PORT, null, null);
                var success = result.AsyncWaitHandle.WaitOne(300);
                if (success)
                {
                    client.EndConnect(result);
                    return true;
                }
                return false;
            }
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// 打开浏览器，带去重保护。
    /// 通过 .browser_opened 标志文件实现：10秒内只开一次。
    /// 通过 .training 标志文件实现：训练期间不开浏览器。
    /// C# 和 Python launcher 共用这些文件。
    /// </summary>
    static bool OpenBrowserIfNeeded(string workDir)
    {
        // 训练中 → 不开浏览器
        if (File.Exists(Path.Combine(workDir, ".training")))
        {
            return false;
        }

        string flagPath = Path.Combine(workDir, BROWSER_FLAG);
        try
        {
            // 当前 Unix 毫秒时间戳
            long nowMs = (long)(DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalMilliseconds;

            // 检查标志文件
            if (File.Exists(flagPath))
            {
                string content = File.ReadAllText(flagPath).Trim();
                long timestamp;
                if (long.TryParse(content, out timestamp))
                {
                    long elapsed = nowMs - timestamp;
                    if (elapsed < BROWSER_COOLDOWN_MS)
                    {
                        // 冷却期内，不开浏览器
                        return false;
                    }
                }
            }

            // 更新标志文件
            File.WriteAllText(flagPath, nowMs.ToString());

            // 打开浏览器
            Process.Start(new ProcessStartInfo(URL) { UseShellExecute = true });
            return true;
        }
        catch
        {
            // 标志文件读写失败时，仍然打开浏览器（降级策略）
            try { Process.Start(new ProcessStartInfo(URL) { UseShellExecute = true }); } catch { }
            return true;
        }
    }

    static void KillPortUsers(string workDir)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "netstat",
                Arguments = "-ano -p TCP",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            using (var p = Process.Start(psi))
            {
                var output = p.StandardOutput.ReadToEnd();
                p.WaitForExit(5000);
                foreach (var line in output.Split('\n'))
                {
                    if (line.Contains(":" + PORT) && line.Contains("LISTENING"))
                    {
                        var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length > 0)
                        {
                            string pid = parts[parts.Length - 1];
                            try
                            {
                                Process.Start(new ProcessStartInfo
                                {
                                    FileName = "taskkill",
                                    Arguments = "/F /PID " + pid,
                                    UseShellExecute = false,
                                    CreateNoWindow = true
                                }).WaitForExit(3000);
                            }
                            catch { }
                        }
                    }
                }
            }
            // 清理浏览器标志和训练标志（杀残留意味着要重新启动）
            try { File.Delete(Path.Combine(workDir, BROWSER_FLAG)); } catch { }
            try { File.Delete(Path.Combine(workDir, ".training")); } catch { }
        }
        catch { }
    }

    static int Main(string[] args)
    {
        string exeDir = AppDomain.CurrentDomain.BaseDirectory;
        string mainExe = Path.Combine(exeDir, "复利引擎.exe");

        // 步骤1：检查后端是否已经在跑
        if (IsServerRunning())
        {
            // 后端在跑 → 尝试开浏览器（带去重保护） → 退出
            OpenBrowserIfNeeded(exeDir);
            return 0;
        }

        // 步骤2：后端没在跑 → 获取 Mutex（我是第一个吗？）
        IntPtr mutex = CreateMutexW(IntPtr.Zero, false, MUTEX_NAME);
        if (mutex == IntPtr.Zero)
        {
            // CreateMutex 失败，直接尝试启动 exe
            if (File.Exists(mainExe))
                Process.Start(new ProcessStartInfo(mainExe) { WorkingDirectory = exeDir, UseShellExecute = false, CreateNoWindow = true });
            return 0;
        }

        bool alreadyExists = (GetLastError() == ERROR_ALREADY_EXISTS);

        if (alreadyExists)
        {
            // Mutex 被占 → 有另一个 exe 正在启动中
            // 最多等 3 秒，让它把服务跑起来
            for (int i = 0; i < 30; i++)
            {
                Thread.Sleep(100);
                if (IsServerRunning())
                {
                    OpenBrowserIfNeeded(exeDir);
                    CloseHandle(mutex);
                    return 0;
                }
            }
            // 3 秒后还没起来，放弃
            CloseHandle(mutex);
            return 0;
        }

        // 步骤3：我是第一个 → 清理残留 → 启动 exe
        KillPortUsers(exeDir);
        Thread.Sleep(200); // 等端口释放

        if (File.Exists(mainExe))
        {
            Process.Start(new ProcessStartInfo(mainExe)
            {
                WorkingDirectory = exeDir,
                UseShellExecute = false,
                CreateNoWindow = true
            });
        }

        CloseHandle(mutex);
        return 0;
    }
}
