foreach ($file in Get-ChildItem -Path ..\..\Dados\Motivation\all_texts) {
    python -m smartedu-summarizer.methods.lexrank `
        -f $file.FullName `
        -m tf-idf `
        -o (Join-Path results lexrank $file.Name) `
        -i ..\..\Dados\CETEMPublico\CETEMPublico_documents `
        -l portuguese `
        -n 1000 `
        -r 0.2 `
        -v
}