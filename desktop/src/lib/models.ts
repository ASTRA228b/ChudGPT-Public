import type { ModelProfile } from "../types";

export const modelProfiles: readonly {
  id: ModelProfile;
  name: string;
  shortName: string;
  description: string;
  family: "public" | "main";
}[] = [
  {
    id: "public",
    name: "ChudGPT-Public V20",
    shortName: "Public V20",
    description: "Independent experimental language model",
    family: "public",
  },
  {
    id: "music",
    name: "ChudGPT-Public-Music V1",
    shortName: "Music V1",
    description: "Original music and lyric model",
    family: "public",
  },
  {
    id: "buggy",
    name: "ChudGPT Buggy",
    shortName: "Buggy",
    description: "Original chaotic personality checkpoint",
    family: "main",
  },
  {
    id: "700",
    name: "ChudGPT 700",
    shortName: "700",
    description: "Legacy personality checkpoint 700",
    family: "main",
  },
  {
    id: "1300",
    name: "ChudGPT 1300",
    shortName: "1300",
    description: "Legacy personality checkpoint 1300",
    family: "main",
  },
  {
    id: "1500",
    name: "ChudGPT 1500",
    shortName: "1500",
    description: "Legacy assistant checkpoint 1500",
    family: "main",
  },
  {
    id: "1600",
    name: "ChudGPT 1600",
    shortName: "1600",
    description: "Legacy assistant checkpoint 1600",
    family: "main",
  },
  {
    id: "ultimate",
    name: "ChudGPT Ultimate",
    shortName: "Ultimate",
    description: "Ultimate checkpoint",
    family: "main",
  },
  {
    id: "plus",
    name: "ChudGPT Plus",
    shortName: "Plus",
    description: "Expanded-context general model",
    family: "main",
  },
  {
    id: "pro",
    name: "ChudGPT Pro",
    shortName: "Pro",
    description: "Longer-form general model",
    family: "main",
  },
  {
    id: "code",
    name: "ChudGPT Code",
    shortName: "Code",
    description: "Coding-focused mode",
    family: "main",
  },
  {
    id: "mega",
    name: "ChudGPT Mega",
    shortName: "Mega",
    description: "Deliberately strange Mega checkpoint",
    family: "main",
  },
] as const;

export const modelProfileIds = modelProfiles.map((model) => model.id);

export function modelProfileInfo(id: ModelProfile) {
  return modelProfiles.find((model) => model.id === id) ?? modelProfiles[0];
}
