import { contextBridge, ipcRenderer } from "electron";

const desktopApi = {
  minimize: () => ipcRenderer.invoke("window:minimize"),
  toggleMaximize: () => ipcRenderer.invoke("window:toggleMaximize"),
  close: () => ipcRenderer.invoke("window:close"),
  isMaximized: () =>
    ipcRenderer.invoke("window:isMaximized") as Promise<boolean>,
  onMaximized: (callback: (maximized: boolean) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, value: boolean) =>
      callback(value);
    ipcRenderer.on("window:maximized", listener);
    return () => {
      ipcRenderer.removeListener("window:maximized", listener);
    };
  },
  loadData: () =>
    ipcRenderer.invoke("data:load") as Promise<Record<string, unknown>>,
  saveData: (value: Record<string, unknown>) =>
    ipcRenderer.invoke("data:save", value) as Promise<boolean>,
  exportData: (value: Record<string, unknown>) =>
    ipcRenderer.invoke("data:export", value) as Promise<boolean>,
  importData: () => ipcRenderer.invoke("data:import") as Promise<unknown>,
  setLaunchAtLogin: (enabled: boolean) =>
    ipcRenderer.invoke("app:setLaunchAtLogin", enabled) as Promise<boolean>,
  getAppInfo: () =>
    ipcRenderer.invoke("app:getInfo") as Promise<{
      version: string;
      platform: string;
    }>,
  openExternal: (url: string) =>
    ipcRenderer.invoke("app:openExternal", url) as Promise<boolean>,
  apiRequest: <T,>(
    endpoint: string,
    method: string,
    body: unknown,
    requestId: string,
  ) =>
    ipcRenderer.invoke(
      "api:request",
      endpoint,
      method,
      body,
      requestId,
    ) as Promise<T>,
  cancelRequest: (requestId: string) =>
    ipcRenderer.invoke("api:cancel", requestId) as Promise<boolean>,
};

contextBridge.exposeInMainWorld("chudDesktop", desktopApi);

export type ChudDesktopApi = typeof desktopApi;
