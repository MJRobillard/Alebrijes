# Fillibuster library

Pre-rendered "side conversation" audio clips the listener can pick, live, to
play while a checkpoint's wrong-answer/chat reply is generating (see
`quiz.html`'s fillibuster dropdown, below the transcript).

Each entry is a `<id>.m4b` with its own `<id>.transcript.json` sidecar,
produced by the normal render pipeline -- write a short script, then:

```
alebrijes cast some-filler-topic.script
alebrijes render some-filler-topic.script some-filler-topic.cast.yaml \
  --out src/alebrijes/fillibusters/some-filler-topic.m4b
```

`list_fillibusters` (`session/fillibuster_library.py`) skips any `.m4b`
that doesn't have a matching `.transcript.json` next to it, so a half-copied
entry never shows up in the picker. This directory ships empty by default.
