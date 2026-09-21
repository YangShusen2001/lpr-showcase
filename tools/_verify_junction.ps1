$ErrorActionPreference = 'Continue'
$out = @{}
$dst = 'C:\Users\26671\Desktop\lpr-showcase'
$src = 'C:\Users\26671\Desktop\车牌识别'

$i = Get-Item -LiteralPath $dst -Force
$out['dst_fullname'] = $i.FullName
$out['dst_attributes'] = $i.Attributes.ToString()
$out['dst_linktype'] = [string]$i.LinkType
$out['dst_target'] = @($i.Target) -join ';'

# 源目录真名（用 .NET 读，避免 PS 编码问题）
$out['src_realpath'] = [System.IO.Path]::GetFullPath($src)
$out['src_exists'] = [System.IO.Directory]::Exists($src)

# 通过联接列目录，确认内容真的转发过来了
$n = (Get-ChildItem -LiteralPath $dst -Force | Measure-Object).Count
$out['entries_via_junction'] = $n

# DevEco 工程（鸿蒙）路径是否纯 ASCII
$deveco = 'C:\Users\26671\lpr-harmony\LprDemo'
$out['deveco_exists'] = [System.IO.Directory]::Exists($deveco)
$out['deveco_is_ascii'] = ($deveco -match '^[\x00-\x7F]+$')
$out['showcase_is_ascii'] = ($dst -match '^[\x00-\x7F]+$')

$json = $out | ConvertTo-Json -Depth 3
[System.IO.File]::WriteAllText("$env:TEMP\junction_verify.json", $json, [System.Text.Encoding]::UTF8)
"written"
