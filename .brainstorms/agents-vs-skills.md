# Agents vs Skills: Mutual Recursion in Claude Code

## The Two Primitives

| | Agent | Skill |
|---|---|---|
| **What it is** | A runtime execution context | A declarative procedure |
| **Definition** | Prompt template + tool permissions + model | Trigger conditions + orchestration logic + document templates + output contracts |
| **Analogy** | A cook | A recipe |
| **Without the other** | Capable but directionless | Inert — just a document |
| **State** | Stateful (holds conversation context, makes judgment calls) | Stateless (defines steps, doesn't execute them) |
| **Cost to define** | Lightweight (a markdown file) | Heavier (trigger rules, I/O contracts, orchestration flow) |

## The Hierarchy: Not Strict Nesting

At first glance, it looks like skills contain agents — a skill like `/autonomous-orchestrator` dispatches multiple agents to do work. But agents can also invoke skills — a subagent can call `/write-spec-docs` or `/write-design-docs` as part of its execution.

This is **mutual recursion**, not a fixed parent-child hierarchy:

```
Skill spawns Agent:    /autonomous-orchestrator → dispatches code-architect agent
Agent invokes Skill:   code-architect agent → calls /write-design-docs
Skill spawns Agent:    /write-design-docs → dispatches a subagent to analyze codebase
```

Each layer can nest the other, to arbitrary depth.

## The One Fixed Root: Main Agent + Main Skill

There is exactly one structural constraint: **the Main Agent is always on top.**

When a user types a message in Claude Code, a Main Agent receives it. That Main Agent is implicitly bound to a "main skill" — the user's intent for the conversation. Everything below that root is mutual recursion.

```
┌─────────────────────────────────────────────┐
│  Main Agent (root — always an agent)        │
│  Bound to user intent (implicit main skill) │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   ┌─────────┐          ┌─────────┐
   │  Skill  │          │  Agent  │
   │  (recipe)│          │  (cook) │
   └────┬────┘          └────┬────┘
        │                    │
   ┌────┴────┐          ┌────┴────┐
   │  Agent  │          │  Skill  │
   │  (cook) │          │  (recipe)│
   └────┬────┘          └────┬────┘
        │                    │
       ...                  ...
   (mutual recursion continues)
```

The root is always an agent because **someone has to hold the thread.** A skill is a document — it can't run itself. It needs an agent to interpret and execute it. So the entry point is always: user → agent → (skill | agent) → ...

## Why This Matters

### 1. Design decisions flow from the root

The Main Agent decides *how* to decompose work. It can:
- Handle it directly (no nesting)
- Invoke a skill (follow a recipe)
- Spawn subagents (delegate to other cooks)
- Invoke a skill that spawns subagents (recipe that needs multiple cooks)

The choice depends on complexity, not on a rigid hierarchy.

### 2. Skills are reusable, agents are contextual

A skill like `/write-spec-docs` can be invoked by:
- The Main Agent directly (user types `/write-spec-docs`)
- An orchestration agent (autonomous-orchestrator dispatches it)
- Another skill's subagent (a pipeline step needs specs first)

The skill doesn't care who called it. The agent calling it provides the context.

### 3. Agents are the connective tissue

When skills chain together (spec → design → implementation → tests), something must hold state between them — deciding what to pass forward, what to retry, what to skip. That connective tissue is always an agent.

```
/autonomous-orchestrator (skill)
  └── orchestrator agent (holds the thread)
        ├── /write-spec-docs (skill) → spec-agent (executes)
        ├── /write-design-docs (skill) → design-agent (executes)
        ├── review gate (agent judgment: proceed or revise?)
        └── /write-implementation-docs (skill) → impl-agent (executes)
```

The orchestrator agent is what makes this a pipeline rather than a pile of documents.

## Summary

| Claim | Why |
|---|---|
| The root is always an agent | Skills can't execute themselves; an agent must interpret them |
| Below the root, it's mutual recursion | Skills spawn agents, agents invoke skills, to arbitrary depth |
| Skills are the investment | They encode reusable procedures, trigger conditions, I/O contracts |
| Agents are cheap but essential | They're "just" prompt + tools + model, but nothing runs without them |
| Agents hold the thread between skills | When skills chain, an agent manages state and judgment between steps |
