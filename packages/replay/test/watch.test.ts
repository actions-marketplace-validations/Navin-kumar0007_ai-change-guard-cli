import type { LlmClient, LlmRequest, LlmResponse } from "@ai-changeguard/llm";
import { describe, expect, it } from "vitest";
import { parseGoldenSet } from "../src/golden.js";
import { createBaseline, formatWatchMarkdown, itemHash, parseBaseline, postSlack, runWatch, slackMessage } from "../src/watch.js";

const SET = parseGoldenSet(
  [
    '{"id":"order","input":"Extract the order as JSON: 2 apples","expect":{"json":true}}',
    '{"id":"greet","input":"Greet Asha in one line","expect":{"contains":["Asha"]}}',
  ].join("\n"),
);

const answer = (text: string, servedModel = "gpt-4o-2024-08-06"): LlmResponse => ({
  text,
  toolCalls: [],
  inputTokens: 10,
  outputTokens: 5,
  latencyMs: 1,
  refused: false,
  servedModel,
});

const isOrder = (req: LlmRequest) => req.messages[0]!.content.startsWith("Extract");
const good = (req: LlmRequest, served?: string) => answer(isOrder(req) ? '{"item": "apple", "qty": 2}' : "Hello Asha, welcome!", served);

/** A model that answers well; `twist` can change the n-th call of a prompt. */
function model(twist?: (req: LlmRequest, n: number) => LlmResponse | undefined): LlmClient {
  const seen = new Map<string, number>();
  return async (req) => {
    const key = req.messages[0]!.content;
    const n = seen.get(key) ?? 0;
    seen.set(key, n + 1);
    return twist?.(req, n) ?? good(req);
  };
}

const NOW = new Date("2026-11-23T03:00:00Z");
const baselineFor = () => createBaseline({ items: SET, model: "openai:gpt-4o", client: model(), runs: 5, concurrency: 1, now: NOW });

describe("baseline", () => {
  it("stores every answer with the model, date and a hash per prompt, and reads back", async () => {
    const b = await baselineFor();
    expect(b).toMatchObject({ version: 1, model: "openai:gpt-4o", createdAt: "2026-11-23T03:00:00.000Z", runs: 5 });
    expect(b.items.map((i) => i.id)).toEqual(["order", "greet"]);
    expect(b.records).toHaveLength(10);
    expect(parseBaseline(JSON.stringify(b))).toEqual(b);
    expect(() => parseBaseline('{"version": 99}')).toThrow(/not an AI Change Guard baseline/);
    expect(() => parseBaseline("nope")).toThrow(/not valid JSON/);
  });
});

describe("watch", () => {
  it("stays quiet when the model answers as before", async () => {
    const w = await runWatch({ items: SET, baseline: await baselineFor(), client: model(), concurrency: 1 });
    expect(w.alert).toBe(false);
    expect(w.report.totals).toMatchObject({ same: 2, changed: 0, broken: 0 });
    expect(formatWatchMarkdown(w)).toContain("AI model watch: no change");
  });

  it("alerts when 3 of 5 answers break, but not for one odd answer", async () => {
    const baseline = await baselineFor();
    const prose = (upTo: number) => (req: LlmRequest, n: number) => (isOrder(req) && n < upTo ? answer("Sure! You ordered 2 apples.") : undefined);
    const broken = await runWatch({ items: SET, baseline, client: model(prose(3)), concurrency: 1 });
    expect(broken.alert).toBe(true);
    expect(broken.report.items.find((i) => i.id === "order")?.verdict).toBe("broken");
    const odd = await runWatch({ items: SET, baseline, client: model(prose(1)), concurrency: 1 });
    expect(odd.alert).toBe(false);
  });

  it("alerts on a reroute to another model and on a new snapshot behind the alias", async () => {
    const baseline = await baselineFor();
    const rerouted = await runWatch({ items: SET, baseline, client: model((req) => good(req, "gpt-5-mini")), concurrency: 1 });
    expect(rerouted.alert).toBe(true);
    expect(rerouted.reroutes).toEqual([{ requested: "gpt-4o", served: "gpt-5-mini" }]);

    const snapshot = await runWatch({ items: SET, baseline, client: model((req) => good(req, "gpt-4o-2024-11-20")), concurrency: 1 });
    expect(snapshot).toMatchObject({ alert: true, reroutes: [], servedBefore: ["gpt-4o-2024-08-06"], servedNow: ["gpt-4o-2024-11-20"] });
    expect(formatWatchMarkdown(snapshot)).toContain("now served by gpt-4o-2024-11-20 (was gpt-4o-2024-08-06)");
  });

  it("leaves prompts edited since the baseline out of the comparison", async () => {
    const baseline = await baselineFor();
    const edited = parseGoldenSet(
      ['{"id":"order","input":"Extract the order as JSON: 3 pears","expect":{"json":true}}', '{"id":"greet","input":"Greet Asha in one line","expect":{"contains":["Asha"]}}'].join("\n"),
    );
    const w = await runWatch({ items: edited, baseline, client: model(), concurrency: 1 });
    expect(w.notInBaseline).toEqual(["order"]);
    expect(w.report.items.map((i) => i.id)).toEqual(["greet"]);
  });
});

describe("Slack", () => {
  it("sends counts and prompt ids, never prompt or answer text, and only to https", async () => {
    const baseline = await baselineFor();
    const w = await runWatch({ items: SET, baseline, client: model((req) => (isOrder(req) ? answer("Sure! 2 apples.") : undefined)), concurrency: 1 });
    const msg = slackMessage(w, "https://github.com/acme/bot/actions/runs/1");
    expect(msg.text).toMatch(/^:rotating_light: openai:gpt-4o changed since the baseline of 2026-11-23: 1 broken\./);
    expect(msg.text).toContain("Prompts: order (broken)");
    expect(msg.text).not.toMatch(/apples|Asha|Extract/);

    let sent: { url: string; body: string } | undefined;
    const fakeFetch = (async (url: string, init: RequestInit) => {
      sent = { url, body: String(init.body) };
      return new Response("ok");
    }) as unknown as typeof fetch;
    await postSlack("https://hooks.slack.com/services/T/B/x", msg, fakeFetch);
    expect(JSON.parse(sent!.body)).toEqual(msg);
    await expect(postSlack("http://example.com/hook", msg, fakeFetch)).rejects.toThrow(/https/);
  });
});

describe("review fixes", () => {
  it("does not alarm every night on an alias the provider always serves under a dated name", async () => {
    const aliased = model((req) => good(req, "claude-3-5-sonnet-20241022"));
    const baseline = await createBaseline({ items: SET, model: "anthropic:claude-3-5-sonnet-latest", client: aliased, runs: 5, concurrency: 1 });
    const w = await runWatch({ items: SET, baseline, client: aliased, concurrency: 1 });
    expect(w).toMatchObject({ alert: false, reroutes: [], snapshotChanged: false });
  });

  it("refuses a baseline in which a prompt got no answers", async () => {
    const limited: LlmClient = async (req) => {
      if (isOrder(req)) throw new Error("HTTP 429: rate limited");
      return good(req);
    };
    await expect(createBaseline({ items: SET, model: "openai:gpt-4o", client: limited, runs: 2, concurrency: 1 })).rejects.toThrow(
      /No answers for order \(HTTP 429: rate limited\)\. If this is a rate limit, try again with a lower --concurrency\./,
    );
  });

  it("fingerprints prompts regardless of key order and ignores the checks", () => {
    const [a] = parseGoldenSet('{"id":"x","input":"hi","tools":[{"name":"t","parameters":{"type":"object","required":["a"]}}],"expect":{"json":true}}');
    const [b] = parseGoldenSet('{"tools":[{"parameters":{"required":["a"],"type":"object"},"name":"t"}],"input":"hi","id":"x"}');
    expect(itemHash(a!)).toBe(itemHash(b!));
    const [c] = parseGoldenSet('{"id":"x","input":"hello"}');
    expect(itemHash(c!)).not.toBe(itemHash(a!));
  });
});

