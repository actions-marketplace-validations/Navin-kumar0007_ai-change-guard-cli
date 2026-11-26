// Starts recording before any of your code runs, so even AI clients created at import time are
// recorded:  node --import @ai-changeguard/recorder/register app.js
// Settings come from the environment: ACG_RECORD_FILE (default ai-calls.jsonl), ACG_RECORD_SAMPLE
// (0 to 1), and ACG_RECORD=0 to turn it off.
import { record } from "./index.js";

const sample = Number(process.env.ACG_RECORD_SAMPLE);
record({
  ...(process.env.ACG_RECORD_FILE ? { file: process.env.ACG_RECORD_FILE } : {}),
  ...(Number.isFinite(sample) && process.env.ACG_RECORD_SAMPLE ? { sample } : {}),
});
