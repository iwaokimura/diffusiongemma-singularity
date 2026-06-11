#!/usr/bin/env python3
"""
DiffusionGemma 推論スクリプト (OpenAI 互換 API クライアント)
Usage:
    # まず別ターミナルでサーバーを起動:
    #   singularity run --nv -B ~/.cache/huggingface:/cache/huggingface diffusiongemma.sif
    #
    # その後このスクリプトを実行:
    python infer.py
    python infer.py --thinking  # Thinking モード有効
    python infer.py --prompt "Prove that sqrt(2) is irrational."
"""

import argparse
from openai import OpenAI


def build_client(base_url: str = "http://localhost:8000/v1") -> OpenAI:
    return OpenAI(base_url=base_url, api_key="dummy")


def generate(
    client: OpenAI,
    prompt: str,
    model: str = "RedHatAI/diffusiongemma-26B-A4B-it-NVFP4",
    max_tokens: int = 512,
    thinking: bool = False,
) -> tuple[str | None, str]:
    messages = [{"role": "user", "content": prompt}]

    # サーバーを --reasoning-parser gemma4 で起動している前提。
    # enable_thinking を渡すと思考部分が reasoning_content として分離されて返る
    extra_body = {}
    if thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning_content", None)
    return reasoning, message.content


def main():
    parser = argparse.ArgumentParser(description="DiffusionGemma 推論クライアント")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="vLLM サーバー URL")
    parser.add_argument("--model", default="RedHatAI/diffusiongemma-26B-A4B-it-NVFP4", help="モデル名")
    parser.add_argument("--prompt", default="Why is the sky blue?", help="プロンプト文字列")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成トークン数")
    parser.add_argument("--thinking", action="store_true", help="Thinking モードを有効化")
    args = parser.parse_args()

    client = build_client(args.url)
    print(f"[*] Model : {args.model}")
    print(f"[*] Prompt: {args.prompt}")
    print(f"[*] Thinking mode: {args.thinking}")
    print("-" * 60)

    reasoning, result = generate(
        client,
        prompt=args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
    )
    if reasoning:
        print("[Thinking]")
        print(reasoning)
        print("-" * 60)
    print(result)


if __name__ == "__main__":
    main()
