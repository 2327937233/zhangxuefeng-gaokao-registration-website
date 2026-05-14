#!/usr/bin/env python3
"""
表达DNA 测试脚本
用法：python test_expression_dna.py

功能：
1. 验证 expression_dna.json 数据完整性
2. 验证 system_prompt.txt 是否包含表达DNA规则
3. 发送测试请求到后端 API，检查回复是否符合张雪峰语气
4. 输出检测报告
"""

import json
import os
import sys


def test_data_file():
    """测试表达DNA数据文件"""
    print("=" * 60)
    print("[1/4] 测试表达DNA数据文件")
    print("=" * 60)

    try:
        from data import expression_dna
        print("[OK] expression_dna.json 加载成功")

        # 检查关键字段
        required_keys = ["scenes", "rhetoric", "vocabulary", "sentence_patterns", "consultation_routines"]
        for key in required_keys:
            if key in expression_dna:
                print(f"  [OK] 模块 '{key}' 存在")
            else:
                print(f"  [FAIL] 模块 '{key}' 缺失")
                return False

        # 检查开场白模板数量
        openings = expression_dna.get("scenes", {}).get("opening", {}).get("templates", [])
        print(f"  [DATA] 开场白模板数量: {len(openings)}")

        # 检查情绪模式数量
        emotions = expression_dna.get("scenes", {}).get("emotion", {}).get("modes", [])
        print(f"  [DATA] 情绪模式数量: {len(emotions)}")
        for mode in emotions:
            print(f"    - {mode.get('emotion')}: {len(mode.get('examples', []))} 个例子")

        # 检查修辞手法
        devices = expression_dna.get("rhetoric", {}).get("devices", [])
        print(f"  [DATA] 修辞手法数量: {len(devices)}")

        # 检查口头禅
        catchphrases = expression_dna.get("vocabulary", {}).get("catchphrases", [])
        print(f"  [DATA] 口头禅数量: {len(catchphrases)}")

        # 检查句式模板
        patterns = expression_dna.get("sentence_patterns", {}).get("patterns", [])
        print(f"  [DATA] 句式模板数量: {len(patterns)}")

        return True

    except Exception as e:
        print(f"[FAIL] 加载失败: {e}")
        return False


def test_system_prompt():
    """测试system prompt是否包含表达DNA规则"""
    print()
    print("=" * 60)
    print("[2/4] 测试 System Prompt 表达DNA规则")
    print("=" * 60)

    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
    if not os.path.exists(prompt_path):
        print(f"[FAIL] 文件不存在: {prompt_path}")
        return False

    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查关键规则是否存在
    checks = [
        ("表达DNA", "表达DNA 章节"),
        ("开场白规则", "开场白规则"),
        ("情绪表达规则", "情绪表达规则"),
        ("转折语", "转折语规则"),
        ("修辞手法", "修辞手法规则"),
        ("高频口头禅", "口头禅规则"),
        ("东北方言", "方言规则"),
        ("句式模板", "句式模板"),
        ("必须有情绪起伏", "情绪起伏规则"),
        ("每200字至少", "频率要求"),
        ("我跟你说", "标志性口头禅"),
        ("没有之一", "绝对化表达"),
    ]

    all_pass = True
    for keyword, label in checks:
        if keyword in content:
            print(f"  [OK] {label}")
        else:
            print(f"  [FAIL] {label} (未找到 '{keyword}')")
            all_pass = False

    return all_pass


def test_api_response():
    """测试API回复是否符合张雪峰语气"""
    print()
    print("=" * 60)
    print("[3/4] 测试 API 回复语气")
    print("=" * 60)

    try:
        import urllib.request
        import urllib.error

        # 获取API base URL
        api_url = "http://localhost:8000/api/consult"

        # 构造测试请求
        test_request = {
            "question": "我孩子想学新闻学专业，靠谱吗？",
            "context": {
                "score": 580,
                "province": "山东",
                "family_background": "普通家庭"
            }
        }

        req = urllib.request.Request(
            api_url,
            data=json.dumps(test_request, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        print(f"  [NET] 发送测试请求到 {api_url}...")
        print(f"  [Q] 测试问题: '{test_request['question']}'")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                answer = data.get("answer", "")
                thinking = data.get("thinking_process", [])

                print(f"  [OK] API 响应成功")
                print(f"  [DATA] 回答长度: {len(answer)} 字")
                print()

                # 检查标志性表达
                print("  [CHK] 语气检测:")

                checks = [
                    ("我跟你说" in answer, "包含'我跟你说'"),
                    ("没有之一" in answer, "包含'没有之一'"),
                    ("？" in answer, "包含反问句"),
                    ("但是" in answer or "问题是" in answer, "包含转折语"),
                    (any(c.isdigit() for c in answer), "包含数字量化"),
                    ("[分析过程]" in answer, "包含分析过程区块"),
                    ("[核心判断]" in answer, "包含核心判断区块"),
                    ("[金句]" in answer, "包含金句区块"),
                    (len(answer) > 100, "回答长度合理(>100字)"),
                ]

                for passed, label in checks:
                    if passed:
                        print(f"    [OK] {label}")
                    else:
                        print(f"    [WARN]  {label}")

                print()
                print("  [OUT] 回答预览（前300字）:")
                print("  " + "-" * 56)
                for line in answer[:300].split("\n"):
                    print(f"  {line}")
                if len(answer) > 300:
                    print("  ...")
                print("  " + "-" * 56)

                return True

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"  [FAIL] API HTTP 错误 {e.code}: {body[:200]}")
            return False
        except urllib.error.URLError as e:
            print(f"  [FAIL] API 连接失败: {e.reason}")
            print(f"  [TIP] 提示: 请确认后端服务已启动 (python main.py 或 uvicorn main:app)")
            return False

    except Exception as e:
        print(f"  [FAIL] 测试异常: {e}")
        return False


def test_consistency():
    """测试system prompt与expression_dna的一致性"""
    print()
    print("=" * 60)
    print("[4/4] 测试 System Prompt 与 Expression DNA 一致性")
    print("=" * 60)

    try:
        from data import expression_dna
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()

        # 检查expression_dna中的口头禅是否都在system prompt中
        catchphrases = expression_dna.get("vocabulary", {}).get("catchphrases", [])
        missing = []
        for item in catchphrases:
            phrase = item.get("phrase", "")
            if phrase and phrase not in prompt:
                missing.append(phrase)

        if missing:
            print(f"  [WARN]  以下口头禅在 system prompt 中未明确提及: {', '.join(missing[:5])}")
        else:
            print("  [OK] 核心口头禅均在 system prompt 中有体现")

        # 检查修辞手法
        devices = expression_dna.get("rhetoric", {}).get("devices", [])
        for device in devices:
            name = device.get("device", "")
            if name in prompt:
                print(f"  [OK] 修辞手法 '{name}' 已融入 system prompt")
            else:
                print(f"  [WARN]  修辞手法 '{name}' 未在 system prompt 中明确命名")

        return True

    except Exception as e:
        print(f"  [FAIL] 测试异常: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  张雪峰表达DNA 测试报告")
    print("=" * 60)

    results = []
    results.append(("数据文件", test_data_file()))
    results.append(("System Prompt", test_system_prompt()))
    results.append(("API 回复", test_api_response()))
    results.append(("一致性", test_consistency()))

    print()
    print("=" * 60)
    print("[测试总结]")
    print("=" * 60)

    for name, passed in results:
        status = "[OK] 通过" if passed else "[FAIL] 失败"
        print(f"  {status} - {name}")

    print()

    # 给出建议
    api_passed = results[2][1] if len(results) > 2 else False
    if not api_passed:
        print("[TIP] 建议:")
        print("  1. 如果 API 测试失败，请确认后端已启动:")
        print("     cd zhanglaoshi && python main.py")
        print("  2. 或者检查 .env 中的 LLM API 配置是否正确")
        print()

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
