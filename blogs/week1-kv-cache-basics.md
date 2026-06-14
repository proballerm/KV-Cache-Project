# Week 1: Understanding KV Cache in LLM Inference

## What Problem Are We Solving?

Large language models generate text one token at a time. Each new token depends on the tokens that came before it.

For example:

User: Tell me a story about a dragon.

The model does not generate the full answer instantly. It generates:

Once  
Once upon  
Once upon a  
Once upon a time  
...

At every step, the model needs to look back at previous tokens.

## What Is Attention?

Attention helps the model decide which previous words matter most when generating the next word.

For example, in the sentence:

The dragon flew over the castle because it was angry.

The model needs attention to understand what "it" refers to.

## What Are Query, Key, and Value?

In transformer models, attention uses three important tensors:

- Query: what the current token is looking for
- Key: information used to match against previous tokens
- Value: information that gets passed forward if the match is useful

## What Is KV Cache?

KV cache stores the Key and Value tensors from previous tokens.

This means when the model generates the next token, it does not need to recompute all previous Key and Value information again.

## Why Does KV Cache Improve Speed?

Without KV cache:

The model repeatedly recomputes attention information for old tokens.

With KV cache:

The model reuses saved Key and Value tensors.

This makes generation faster, especially for long conversations.

## Why Is This Useful for Games?

In a real time game, players expect fast responses from NPCs.

If an AI character takes too long to respond, the game feels slow.

KV cache can help make AI NPC conversations faster and smoother.

## Week 1 Goal

This week, we will build a simple Python script that runs text generation and measures response latency.