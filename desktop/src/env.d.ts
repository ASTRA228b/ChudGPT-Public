/// <reference types="vite/client" />

import type { ChudDesktopApi } from "../electron/preload";

declare global {
  interface Window {
    chudDesktop: ChudDesktopApi;
  }
}

export {};
