from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

logger = logging.getLogger('smartedu-summarizer')

def pegasus_abs(text):

    model = "google/pegasus-cnn_dailymail"

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)

    inputs = tokenizer(text, return_tensors='pt', truncation=True)

    # Generate Summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, early_stopping=True)

    summary_generated = ""
    for g in summary_ids:
        tok = tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        summary_generated += tok

    summary_generated = summary_generated.replace("<n>", " ")
    return summary_generated
    

def summarize(args):
    logger.info(f'Loading file \'{args.file}\'')
    with args.file.open('rU', encoding='utf-8') as file_:
        text = file_.read()
    logger.info(f'File loaded')
    logger.debug(f'Loaded text\n\n{text}')

    logger.info('Summarizing')
    summary = pegasus_abs(text)
    logger.info('Summarization done')
    logger.debug(f'Summary\n\n{summary}')

    logger.info('Saving summary')
    with args.output.open('w', encoding='utf-8') as file_:
        file_.write(summary)
    logger.info(f'Saved summary to \'{args.output}\'')