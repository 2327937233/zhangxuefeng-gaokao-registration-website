import re

with open('C:/Users/23279/Desktop/zhanglaoshi/zhiyuan-agent.html', 'r', encoding='utf-8') as f:
    content = f.read()

issues = []

# 1. Check for unclosed JSX comments
jsx_expr_opens = content.count('{/*')
jsx_expr_closes = content.count('*/}')
if jsx_expr_opens != jsx_expr_closes:
    issues.append(f'JSX comment mismatch: {jsx_expr_opens} open, {jsx_expr_closes} close')

# 2. Check for template literals
template_literals = []
for match in re.finditer(r'`([^`]*\$\{[^}]*\}[^`]*)*`', content):
    template_literals.append(match.group(0)[:50])
print(f'Template literals found: {len(template_literals)}')

# 3. Check for lines with className and backticks
lines = content.split('\n')
for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if 'className=' in stripped and '`' in stripped:
        issues.append(f'Line {i}: className with backtick - potential issue')

# 4. Check for invalid JSX attribute names
invalid_attrs = re.findall(r'\s([a-z]+-[a-z]+)=', content)
invalid_jsx_attrs = [a for a in invalid_attrs if not a.startswith('data-') and not a.startswith('aria-') and not a.startswith('x-')]
if invalid_jsx_attrs:
    issues.append(f'Potential invalid JSX attributes: {set(invalid_jsx_attrs)}')

# 5. Check for JSX expressions that return arrays without key
jsx_arrays = re.findall(r'\{[\s\S]*?\.map\s*\(', content)
print(f'Map expressions in JSX: {len(jsx_arrays)}')

if issues:
    print('Potential issues found:')
    for issue in issues:
        print(f'  - {issue}')
else:
    print('No obvious issues found in static analysis')
