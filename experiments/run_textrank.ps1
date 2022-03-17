foreach ($file in Get-ChildItem -Path ..\..\Dados\Motivation\all_texts) {
    python -m smartedu-summarizer.methods.textrank `
        -f $file.FullName `
        -o (Join-Path results textrank $file.Name) `
        -l portuguese `
        -r 0.2 `
        -v
}