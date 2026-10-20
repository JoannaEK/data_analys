$out = "C:\Users\ПРОСТО\Documents\Проект по умолчанию\n4_1-1000.csv"
"n,n^4" | Out-File -FilePath $out -Encoding utf8
for ($i = 1; $i -le 1000; $i++) {
  $v = [int64]($i*$i*$i*$i)
  "$i,$v" | Out-File -FilePath $out -Append -Encoding utf8
}
Write-Host "done"
