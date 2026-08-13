import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  nativeTheme,
  shell,
} from "electron";
import Store from "electron-store";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const API_BASE = "https://chudgpt-public.vercel.app";
const ALLOWED_LINKS = new Set([
  "https://chudgpt-public.vercel.app/",
  "https://chudgpt-landing.vercel.app/",
  "https://github.com/ASTRA228b/ChudGPT-Public",
]);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataStore = new Store<Record<string, unknown>>({
  name: "chudgpt-desktop-data",
});
const requests = new Map<string, AbortController>();
let mainWindow: BrowserWindow | null = null;

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1360,
    height: 860,
    minWidth: 900,
    minHeight: 620,
    frame: false,
    show: false,
    backgroundColor: "#05080f",
    title: "ChudGPT Desktop",
    icon: path.join(__dirname, "../assets/icon.png"),
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.once("ready-to-show", () => mainWindow?.show());
  mainWindow.on("maximize", () =>
    mainWindow?.webContents.send("window:maximized", true),
  );
  mainWindow.on("unmaximize", () =>
    mainWindow?.webContents.send("window:maximized", false),
  );
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (ALLOWED_LINKS.has(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (event, url) => {
    const localUrl = mainWindow?.webContents.getURL();
    if (localUrl && url !== localUrl) event.preventDefault();
  });

  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl) void mainWindow.loadURL(devUrl);
  else void mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
}

async function requestApi(
  endpoint: "status" | "chat" | "generate" | "clear",
  method: "GET" | "POST",
  body: unknown,
  requestId: string,
): Promise<unknown> {
  const controller = new AbortController();
  requests.set(requestId, controller);
  const timeout = setTimeout(
    () => controller.abort("timeout"),
    endpoint === "status" ? 12_000 : 90_000,
  );
  try {
    const response = await fetch(`${API_BASE}/api/${endpoint}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: method === "POST" ? JSON.stringify(body ?? {}) : undefined,
      signal: controller.signal,
    });
    const raw = await response.text();
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      throw new Error("ChudGPT-Public returned a malformed response.");
    }
    if (!response.ok) {
      const value = parsed as { error?: string; detail?: string };
      throw new Error(
        value.error ||
          value.detail ||
          `ChudGPT-Public returned HTTP ${response.status}.`,
      );
    }
    return parsed;
  } catch (error) {
    if (controller.signal.aborted)
      throw new Error("The ChudGPT-Public request timed out or was cancelled.");
    throw error instanceof Error
      ? error
      : new Error("Could not connect to ChudGPT-Public.");
  } finally {
    clearTimeout(timeout);
    requests.delete(requestId);
  }
}

function registerIpc(): void {
  ipcMain.handle("window:minimize", () => mainWindow?.minimize());
  ipcMain.handle("window:toggleMaximize", () => {
    if (mainWindow?.isMaximized()) mainWindow.unmaximize();
    else mainWindow?.maximize();
  });
  ipcMain.handle("window:close", () => mainWindow?.close());
  ipcMain.handle(
    "window:isMaximized",
    () => mainWindow?.isMaximized() ?? false,
  );
  ipcMain.handle("data:load", () => dataStore.store);
  ipcMain.handle("data:save", (_event, value: Record<string, unknown>) => {
    dataStore.store = value;
    return true;
  });
  ipcMain.handle(
    "data:export",
    async (_event, value: Record<string, unknown>) => {
      const result = await dialog.showSaveDialog(mainWindow!, {
        title: "Export ChudGPT chats",
        defaultPath: `ChudGPT-Desktop-export-${new Date().toISOString().slice(0, 10)}.json`,
        filters: [{ name: "JSON", extensions: ["json"] }],
      });
      if (result.canceled || !result.filePath) return false;
      await writeFile(result.filePath, JSON.stringify(value, null, 2), "utf8");
      return true;
    },
  );
  ipcMain.handle("data:import", async () => {
    const result = await dialog.showOpenDialog(mainWindow!, {
      title: "Import ChudGPT chats",
      properties: ["openFile"],
      filters: [{ name: "JSON", extensions: ["json"] }],
    });
    if (result.canceled || !result.filePaths[0]) return null;
    const raw = await readFile(result.filePaths[0], "utf8");
    if (raw.length > 10_000_000)
      throw new Error("Import is larger than the 10 MB safety limit.");
    return JSON.parse(raw) as unknown;
  });
  ipcMain.handle("app:setLaunchAtLogin", (_event, enabled: boolean) => {
    app.setLoginItemSettings({ openAtLogin: enabled, path: process.execPath });
    return app.getLoginItemSettings().openAtLogin;
  });
  ipcMain.handle("app:getInfo", () => ({
    version: app.getVersion(),
    platform: process.platform,
  }));
  ipcMain.handle("app:openExternal", async (_event, url: string) => {
    if (!ALLOWED_LINKS.has(url))
      throw new Error("That external address is not allowed.");
    await shell.openExternal(url);
    return true;
  });
  ipcMain.handle("api:request", (_event, endpoint, method, body, requestId) =>
    requestApi(endpoint, method, body, requestId),
  );
  ipcMain.handle("api:cancel", (_event, requestId: string) => {
    requests.get(requestId)?.abort("cancelled");
    return true;
  });
}

app.whenReady().then(() => {
  nativeTheme.themeSource = "dark";
  registerIpc();
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
