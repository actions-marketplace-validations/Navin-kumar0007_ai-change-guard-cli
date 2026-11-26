# AI Change Guard recorder

Records your app's real AI calls to a JSONL file. `ai-change-guard import` turns that file into a golden set, so you test a model switch on real prompts without writing any.

## Node.js / TypeScript (three lines)

The recorder is not on npm yet (it will be at launch). Until then, copy [`src/index.ts`](src/index.ts) into your project as `acg-recorder.ts`: one file, no dependencies. Then:

```ts
import { record } from "./acg-recorder.js";

record(); // writes ./ai-calls.jsonl
```

It wraps `fetch`, which the official OpenAI, Anthropic and Google Gen AI SDKs use (Node 18 or newer). Call it before you create your AI clients: a client created earlier may keep the original `fetch` and is then not recorded. The safest place is the first line of your entry file, in a module of its own that you import first. Once the package is published, `node --import @ai-changeguard/recorder/register app.js` starts it before any of your code (settings from `ACG_RECORD_FILE` and `ACG_RECORD_SAMPLE`).

## Python (three lines)

Copy [`python/acg_recorder.py`](python/acg_recorder.py) into your project (one file, no dependencies of its own). It wraps `httpx` and `httpx2`, which the OpenAI, Anthropic and Google Gen AI SDKs already use (google-genai 3.x sends through `httpx2`). Then:

```python
from acg_recorder import record

record()  # writes ./ai-calls.jsonl
```

## Then

```
ai-change-guard import ai-calls.jsonl --out golden.jsonl
```

## What it records, and what it never does

- Only POST calls to known AI APIs: OpenAI, Anthropic, Gemini, Vertex AI, Azure OpenAI, OpenRouter, Groq, Mistral, Together and DeepSeek. Add your own gateway with `hosts: ["llm.mycompany.internal"]`.
- For each successful call: the time, the URL without its query string, the model, the HTTP status, and the JSON request and response bodies. Calls the provider rejected (4xx/5xx) are left out. For a streamed answer only the request is kept, and the stream is left untouched for your code.
- It never records headers, so API keys are never written, and never the URL query (Gemini can carry `?key=` there).
- Tested live with the real `@google/genai` (Node) and `google-genai` (Python) SDKs.
- It never changes what your app sends or receives, and any recorder error is swallowed, never raised in your code.
- The file is capped (20 MB by default, `maxBytes` / `max_bytes`); recording stops when it is full.

The file holds your users' real messages. Keep it out of git, and use `redact` to remove or replace anything private before it is written:

```ts
record({
  sample: 0.2, // keep 1 call in 5
  redact: (call) => (JSON.stringify(call.request).includes("password") ? null : call),
});
```

```python
record(sample=0.2, redact=lambda call: None if "password" in str(call["request"]) else call)
```

Set `ACG_RECORD=0` to turn recording off without changing code, for example in production.
