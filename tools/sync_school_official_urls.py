import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def normalize_url(url: str) -> str:
    url = url.strip().rstrip("。；;，,")
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def parse_official_urls() -> dict[str, str]:
    urls: dict[str, str] = {}
    sources = [
        ROOT / "双一流高校官网地址.txt",
        ROOT / "非双一流高校地址.txt",
    ]

    markdown_row = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(?:\[)?(https?://[^)\]\s|]+)")
    plain_row = re.compile(r"^([^：:\s][^：:]+?)\s*[：:]\s*(https?://\S+)")

    for source in sources:
        text = source.read_text(encoding="utf-8")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("##", "| --", "|--")):
                continue

            match = markdown_row.search(line) or plain_row.search(line)
            if not match:
                continue

            school = match.group(1).strip()
            if school in {"高校", "官网"} or " " in school:
                continue
            url = normalize_url(match.group(2))
            if url:
                urls[school] = url

    return dict(sorted(urls.items(), key=lambda item: item[0]))


def update_schools_json(urls: dict[str, str]) -> int:
    target = ROOT / "data" / "schools.json"
    schools = json.loads(target.read_text(encoding="utf-8"))
    matched = 0
    for school in schools:
        official_url = urls.get(school["name"])
        if official_url:
            school["official_url"] = official_url
            matched += 1
        else:
            school.pop("official_url", None)
    target.write_text(json.dumps(schools, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return matched


def update_frontend(urls: dict[str, str]) -> None:
    target = ROOT / "zhiyuan-agent.html"
    html = target.read_text(encoding="utf-8")
    mapping = json.dumps(urls, ensure_ascii=False, indent=16)
    block = f"            const schoolOfficialUrls = {mapping};\n\n"

    existing = re.compile(r"\n\s*const schoolOfficialUrls = \{.*?\};\n", re.S)
    if existing.search(html):
        html = existing.sub("\n" + block, html)
    else:
        marker = "            const badgeStyles = {"
        html = html.replace(marker, block + marker, 1)

    old_card = """                                                return (
                                                    <div
                                                        key={uni}
                                                        className="bg-white border border-ink-100 rounded-xl p-4 hover-lift cursor-pointer card-accent-bar relative group"
                                                    >
                                                        <div className="text-sm font-medium text-ink-900">{uni}</div>
                                                        {badge && (
                                                            <span className={`inline-block mt-2 px-2 py-0.5 ${badgeStyles[badge]} text-[10px] font-bold rounded border`}>
                                                                {badge}
                                                            </span>
                                                        )}
                                                    </div>
                                                );"""

    new_card = """                                                const officialUrl = schoolOfficialUrls[uni];
                                                const CardTag = officialUrl ? 'a' : 'div';
                                                return (
                                                    <CardTag
                                                        key={uni}
                                                        href={officialUrl || undefined}
                                                        target={officialUrl ? "_blank" : undefined}
                                                        rel={officialUrl ? "noopener noreferrer" : undefined}
                                                        title={officialUrl ? `打开${uni}官网` : `${uni}暂无官网地址`}
                                                        className={`bg-white border border-ink-100 rounded-xl p-4 hover-lift card-accent-bar relative group block transition-colors ${officialUrl ? 'cursor-pointer hover:border-zxf-red/30' : 'cursor-default opacity-80'}`}
                                                    >
                                                        <div className="flex items-start justify-between gap-2">
                                                            <div className="text-sm font-medium text-ink-900">{uni}</div>
                                                            {officialUrl && <Icon name="external-link" size={14} className="text-ink-300 group-hover:text-zxf-red flex-shrink-0" />}
                                                        </div>
                                                        {badge && (
                                                            <span className={`inline-block mt-2 px-2 py-0.5 ${badgeStyles[badge]} text-[10px] font-bold rounded border`}>
                                                                {badge}
                                                            </span>
                                                        )}
                                                    </CardTag>
                                                );"""

    if old_card not in html:
        raise RuntimeError("Could not find university card block in zhiyuan-agent.html")
    html = html.replace(old_card, new_card, 1)
    target.write_text(html, encoding="utf-8")


def main() -> None:
    urls = parse_official_urls()
    matched = update_schools_json(urls)
    update_frontend(urls)
    (ROOT / "school_official_urls.json").write_text(
        json.dumps(urls, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Parsed {len(urls)} official urls; matched {matched} backend schools.")


if __name__ == "__main__":
    main()
