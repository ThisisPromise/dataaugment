import argparse
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBartTokenizerFast, AutoModelForSeq2SeqLM, AutoTokenizer

def load_model_and_tokenizer(model_name):
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBartTokenizerFast.from_pretrained(model_name)
    return model, tokenizer

def load_nllb_model_and_tokenizer(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def perform_translation(model, tokenizer, batch_texts):
    formatted_batch_texts = [f"{text}" for text in batch_texts]
    model_inputs = tokenizer(formatted_batch_texts, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**model_inputs)
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_texts

def perform_back_translation(original_texts, source_lang, target_lang, original_model, original_tokenizer, back_translation_model, back_translation_tokenizer):
    temp_translated_batch = perform_translation(original_model, original_tokenizer, original_texts)
    back_translated_batch = perform_translation(back_translation_model, back_translation_tokenizer, temp_translated_batch)
    return list(set(original_texts) | set(back_translated_batch))

def main():
    parser = argparse.ArgumentParser(description="Perform translation and back-translation with MBART and NLLB models.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for tokenizer and model loading.")
    parser.add_argument("--back_translation_model_name", type=str, required=True, help="Model name for back translation.")
    parser.add_argument("--source_lang", type=str, required=True, help="Source language code.")
    parser.add_argument("--target_lang", type=str, required=True, help="Target language code.")
    
    args = parser.parse_args()

    # Load Mafan dataset
    dataset = load_dataset('masakhane/mafand', args.source_lang + '-' + args.target_lang)
    original_texts = dataset['train']['translation']['en']  

    # Load models and tokenizer
    original_model, original_tokenizer = load_model_and_tokenizer(args.model_name)
    back_translation_model, back_translation_tokenizer = load_nllb_model_and_tokenizer(args.back_translation_model_name)

    # Perform translation
    translated_texts = perform_translation(original_model, original_tokenizer, original_texts[:5])  # Limiting to 5 for demonstration
    print("Translated Texts:", translated_texts)

    # Perform back-translation
    back_translated_texts = perform_back_translation(original_texts[:5], args.source_lang, args.target_lang, original_model, original_tokenizer, back_translation_model, back_translation_tokenizer)
    print("Back-Translated Texts:", back_translated_texts)

if __name__ == "__main__":
    main()
