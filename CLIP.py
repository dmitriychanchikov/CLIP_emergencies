# Импорты для обработки изображений
import cv2
import numpy as np
from PIL import Image

# Импорты для работы с торчем
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Импорты для работы с файлами и системой
import os
import requests
import subprocess

# Импорты для работы с текстом и документами
import ftfy
import gzip
import html
import regex as re
from docx import Document
from docx.shared import Cm
from functools import lru_cache

print("Torch version:", torch.__version__)


# Загрузка модели CLIP
def load_model(model_path):
    """
    Скачиваем CLIP, предобученный на 400М пар изображение-текст.  
    Его можно использовать в режиме обучения без обучения (например ViT-B/32 CLIP). 
    После запуска блока нас ждет установка скачивание model.pt модели CLIP: Visual Transformer "ViT-B/32" + Text Transformer
    """
    if not os.path.exists(model_path):
        url_model = {"ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"}
        subprocess.run(["curl", "-o", model_path, url_model["ViT-B/32"]])

    model = torch.jit.load(model_path).cuda().eval()
    input_resolution = model.input_resolution.item()
    context_length = model.context_length.item()
    vocab_size = model.vocab_size.item()

    print("CLIP (ViT-B/32)")
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}".replace(',', "'"))
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    return model


# Препроцессинг Текста
def load_vocab(vocab_path):
    """
    Текстовый препроцессинг для Text Transformer части сети CLIP использует нечувствительный к регистру токенизатор.
    Код токенизатора скрыт во второй ячейке блока. Далее текст паддится до длины сontext length, и готов подаваться в трансформер.
    """
    if not os.path.exists(vocab_path):
        url_bpe = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
        subprocess.run(["curl", "-o", vocab_path, url_bpe])

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        print(self)

    def __str__(self):
            return (
                f"SimpleTokenizer\n"
                f"Byte encoder: {len(self.byte_encoder)}\n"
                f"Byte decoder: {len(self.byte_decoder)}\n"
                f"Merges: {len(self.bpe_ranks)}\n"
                f"Vocab size: {len(self.encoder)}\n"
            )
    
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)
        if not pairs:
            return token+'</w>'
        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


# # Препроцессинг Изображений
# def preprocess_image(image_path):  # переписать под случаи, когда подается уже загруженное изображение
#     """
#     Так как модель CLIP представляет из себя Visual Transformer "ViT-B/32", это означает, 
#     что вход модели должен быть фиксированного разрешения 224x224 пикселя. 
#     Препроцессинг изображений представляет из себя изменение размера входного изображения и обрезка его по центру. 
#     Перед этим нормализуем яркость пикселей картинки, 
#     используя поканальное  среднее значение датасета 400М пар изображение-текст и стандартное отклонение.
#     """
#     preprocess = Compose([Resize((224,224), interpolation=Image.BICUBIC), ToTensor()])
#     image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
#     image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

#     image = Image.open(image_path).convert('RGB')
#     image = preprocess(image)

#     print(image.shape)
#     if image.shape[0] != 3:
#         raise TypeError("Only 3-channel RGB image are allowed")
    
#     image = torch.tensor(np.stack(image)).cuda()
#     image -= image_mean[:, None, None]
#     image /= image_std[:, None, None]

#     return image


# # Классификации объектов на изображении
# def predict_CLIP(model, tokenizer, image, text_descriptions):
#     sot_token = tokenizer.encoder['<|startoftext|>']
#     eot_token = tokenizer.encoder['<|endoftext|>']

#     text_tokens = [[sot_token] + tokenizer.encode(desc) + [eot_token] for desc in text_descriptions]
#     text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)

#     for i, tokens in enumerate(text_tokens):
#         text_input[i, :len(tokens)] = torch.tensor(tokens)

#     text_input = text_input.cuda()
#     text_input.shape

#     image = image.unsqueeze(0)
#     with torch.no_grad():
#         image_features = model.encode_image(image).float()
#         image_features /= image_features.norm(dim=-1, keepdim=True) # 512 -> 256 -> 1 (1/0) (N -> 512)
#         text_features = model.encode_text(text_input).float()
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#         text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#         top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
        
#     return top_labels, top_probs

