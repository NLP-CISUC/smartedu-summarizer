foreach ($file in Get-ChildItem ..\..\Dados\Motivation\all_texts) {
    python -m smartedu-summarizer.methods.lexrank `
        -f $file.FullName `
        -m bert `
        -o (Join-Path results lexrank-bert $file.Name) `
        -v
}