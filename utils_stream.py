from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

from CLIP import *


def time_to_str(time: float) -> str:
    """
    Преобразует время в формат строки

    Аргументы:
        time (float): время в секундах

    Возвращает:
        str: cтроковое представление времени в формате 'mm:ss.sss'
    """

    minutes = time // 60
    seconds = time % 60
    res = f'{minutes:02.0f}:{seconds:06.3f}'

    return res


def preprocess_text(model: torch.nn.Module, tokenizer: SimpleTokenizer, text_descriptions: List[str], add_word: str = None) -> Dict[str, torch.Tensor]:
    """
    Кодирует текстовые описания с использованием токенизатора

    Аргументы:
        model (torch.nn.Module): экземпляр модели CLIP
        tokenizer (SimpleTokenizer): экземпляр токенизатора
        text_descriptions (List[str]): список текстовых описаний
        add_word (str): слово для добавления к описанию (по умолчанию add_word=None)

    Возвращает:
        Dict[str, torch.Tensor]: cловарь, где ключами являются text_descriptions, а значениями - их тензоры PyTorch
    """

    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']
    text_tokens_dict = {}

    for term in text_descriptions:
        text = f'A {" ".join([add_word, "photo of"] if add_word else ["photo of"])} {term}'

        term_tokens = [sot_token] + tokenizer.encode(text) + [eot_token]
        text_tokens = torch.zeros(model.context_length, dtype=torch.long)
        text_tokens[:len(term_tokens)] = torch.tensor(term_tokens)
        text_tokens = text_tokens.cuda()

        with torch.no_grad():
            text_tensor = text_tokens.unsqueeze(0)
            text_features = model.encode_text(text_tensor).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

        key = f'({add_word[:4]}) {term}' if add_word else f'{term}'
        text_tokens_dict[key] = text_features

    return text_tokens_dict


def preprocess_image(model: torch.nn.Module, image: Image.Image, preprocess: Compose) -> torch.Tensor:
    """
    Предобрабатывает изображение для модели CLIP

    Аргументы:
        image: RGB изображение в формате объекта PIL.Image.Image
        preprocess: объект Compose для предобработки изображения

    Возвращает:
        Тензор PyTorch изображения
    """

    # Приведение изображения к RGB формату PIL.Image.Image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Применение предобработки
    image = preprocess(image)

    # Проверка на 3-канальность RGB изображения
    if image.shape[0] != 3:
        raise TypeError('Допускается только З-канальное RGB изображение')

    # Получение констант предобработки
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    
    # Приведение изображения к тензору PyTorch и нормализация
    image = torch.from_numpy(np.array(image)).cuda()
    image -= image_mean[:, None, None]
    image /= image_std[:, None, None]

    # 
    with torch.no_grad():
        image_tensor = image.unsqueeze(0)
        image_features = model.encode_image(image_tensor).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features


def process_video(model: torch.nn.Module, text_tensor_dict: Dict[str, torch.Tensor], video_path: str, fps: Union[int, float]) -> pd.DataFrame:
    """
    Обрабатывает видеофайл, вычисляя косинусное сходство между кадрами и текстовыми описаниями

    Аргументы:
        model (torch.nn.Module): экземпляр модели CLIP
        text_tensor_dict (Dict[str, torch.Tensor]): cловарь, где ключами являются текстовые описания, а значениями - их тензоры PyTorch
        video_path (str): источник видео
        fps (Union[int, float]): частота кадров в секунду

    Возвращает:
        pd.DataFrame: DataFrame, содержащий значения косинусного сходства для каждого кадра видеофайла и текстового описания
    """

    # Проверка числа fps
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError('Частота кадров (fps) должна быть больше нуля')

    try:
        # Открытие видеофайла
        video_capture = cv2.VideoCapture(video_path)

        # Проверка успешности открытия видеофайла
        if not video_capture.isOpened():
            raise FileNotFoundError(f'Файл {video_path} не найден или не может быть прочитан')

        # Вычисление количества кадров, которые нужно пропустить для заданного fps
        total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        skip_frames = frame_rate / fps

        # Вывод информации о видеофайле
        full_time = time_to_str(total_frames/frame_rate)
        print(f'Файл: {video_path}, продолжительность: {full_time}, частота кадров (fps): {frame_rate}')
        
        # Создание объекта Compose для предобработки изображения
        preprocess = Compose([Resize((224,224), interpolation=Image.BICUBIC), ToTensor()])

        # Создание пустого датафрейма для хранения значений косинусного сходства
        df = pd.DataFrame(columns=list(text_tensor_dict.keys()))
        
        # Итеративное извлечение нужных кадров
        for frame_number in range(0, int(total_frames), int(skip_frames)):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video_capture.read()

            # Проверка успешности извлечения кадра
            if not ret:
                raise RuntimeError('Ошибка при извлечении кадра')

            # Преобразование кадра в нужный формат 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = preprocess_image(model, frame, preprocess)

            # Расчет времени текущего кадра
            current_time = time_to_str(frame_number / frame_rate)

            frame_data = {}
            for text, text_tensor in text_tensor_dict.items():
                cos_sim = F.cosine_similarity(frame_tensor, text_tensor)
                frame_data[text] = cos_sim.item()

            frame_df = pd.DataFrame(frame_data, index=[current_time])
            df = pd.concat([df, frame_df])

    except Exception as e:
        # Вывод сообщения об ошибке
        print(f"Ошибка: {e}")
    
    finally:
        # Закрытие видеофайла в любом случае
        if 'video_capture' in locals() and video_capture.isOpened():
            video_capture.release()

    return df


def save_df(df: pd.DataFrame, format: str, save_path: str, video_path: str, fps: Union[int, float], text_len: int, add_word: str = None) -> None:
    """
    Сохраняет DataFrame с результатами в формате 'csv' или 'excel' в заданной директории

    Аргументы:
        df (pd.DataFrame): сохраняемый DataFrame
        format (str): формат сохранения данных ('csv' или 'excel').
        save_path (str): путь к директории для сохранения
        video_path (str): путь к видеофайлу (используется для формирования имени сохраняемого файла)
        fps (Union[int, float]): частота кадров видеофайла
        text_len (int): Количество текстовых описаний (столбцов DataFrame)
        add_word (str, optional): дополнительное слово для формирования имени файла (по умолчанию add_word = None)
    """
    
    # Формирование пути и имени сохраняемого файла
    dir, file = video_path.split('/')[-2:]
    save_dir = f'{save_path}/{dir}'
    basename = os.path.splitext(os.path.basename(file))[0]
    if add_word is not None:
        name = f'{basename}_{fps}fps_{text_len}td_{add_word[:4]}'
    else:
        name = f'{basename}_{fps}fps_{text_len}td'

    # Создание директории и сохранение DataFrame в указанном формате
    os.makedirs(save_dir, exist_ok=True)
    if format == 'csv':
        df.to_csv(f'{save_dir}/{name}.csv')
        print(f'Сохранено в {save_dir}/{name}.csv')
    elif format == 'excel':
        df.to_excel(f'{save_dir}/{name}.xlsx')
        print(f'Сохранено в {save_dir}/{name}.xlsx')
    else:
        raise ValueError("Неподдерживаемый формат. Допустимые значения: 'csv' или 'excel'")


def plot_results(df: pd.DataFrame, xticks: int, plots: int = 3) -> None:
    """
    Строит графики для результатов из DataFrame

    Аргументы:
        df (pd.DataFrame): DataFrame с результатами
        xticks (int): количество отметок времени на горизонтальной оси
        plots (int, optional): количество графиков, отображаемых на одной фигуре (по умолчанию plots=3)
    """

    for i in range(0, len(df.columns), plots):
        plt.figure(figsize=(15, 4))

        for j in range(plots):
            if i + j < len(df.columns):
                plt.plot(df.index, df.iloc[:, i + j], label=df.columns[i + j])

        plt.xticks(df.index[::xticks], rotation=90)
        plt.ylim((0.15, 0.3))
        plt.legend()
        plt.show()
