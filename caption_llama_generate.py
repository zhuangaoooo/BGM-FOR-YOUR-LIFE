import torch
from transformers import AutoConfig, AutoModel
from transformers import pipeline
from modelscope import snapshot_download
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu 
import scipy
from diffusers import AudioLDM2Pipeline
import os
import time
import cv2
os.environ["TRANSFORMERS_OFFLINE"] = "1"
def image_captioning(image):
    model_path = 'mPLUG/mPLUG-Owl3-2B-241014'
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half, trust_remote_code=True)
    processor = model.init_processor(tokenizer)
    model.to('cuda')
    # 指定保存路径


    messages = [
        {"role": "user", "content": """<|image|>
    Please describe the location and the action I am performing in one sentence each."""},
        {"role": "assistant", "content": "Location: [location]. Action: [action]."}
    ]

    inputs = processor(messages, images=[image], videos=None)

    inputs.to('cuda')
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens':100,
        'decode_text':True,
    })


    g = model.generate(**inputs)
    print(g[0])
    return g

def llama_1(infor_image,pulse,location):
    model_dir = r'C:\Users\张博伦\.cache\modelscope\hub\LLM-Research\Llama-3___2-1B-Instruct'

    pipe = pipeline(
        "text-generation",
        model=model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Help me generate music prompt suitable for the location and action they are in one sentence."+infor_image[0]}
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])
    return outputs[0]["generated_text"][-1]



def llama(infor_image,pulse,location):
    model_dir = r'C:\Users\张博伦\.cache\modelscope\hub\LLM-Research\Llama-3___2-1B-Instruct'

    pipe = pipeline(
        "text-generation",
        model=model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(infor_image)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Choose the best music style and instruments according to the next sentence."+infor_image}
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])
    return outputs[0]["generated_text"][-1]

def musicgen(prompt,audio_name):
    # 确保 PyTorch 可以使用 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建 pipeline 并指定使用 GPU
    synthesiser = pipeline("text-to-audio", model="facebook/musicgen-small",device='cuda')


    # 生成音乐
    music = synthesiser(prompt, forward_params={"do_sample": True})

    scipy.io.wavfile.write('wav/'+str(audio_name)+'.wav', rate=music["sampling_rate"], data=music["audio"])
'''
def AudioLDM(prompt,audio_name):
    # load the pipeline
    torch.cuda.empty_cache()
    repo_id = "cvssp/audioldm2-music"
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # define the prompts
    #prompt = "incorporating soft piano and acoustic guitar elements to evoke a sense of quiet contemplation, set against the subtle sounds of a gentle breeze and distant city hum, reminiscent of a peaceful office or study space with a window in the background"
    negative_prompt = "Low quality."

    # set the seed
    generator = torch.Generator("cuda").manual_seed(0)

    # run the generation
    audio = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=100,
        audio_length_in_s=20.0,
        num_waveforms_per_prompt=10,
    ).audios

    # save the best audio sample (index 0) as a .wav file
    scipy.io.wavfile.write(str(audio_name)+'.wav', rate=16000, data=audio[0])
'''

def capture_picture_from_esp32_cam(url):
    # 持续尝试打开视频流，直到成功
    cap = cv2.VideoCapture(url)
    while not cap.isOpened():
        if not cap.isOpened():
            print("Failed to open video stream, trying again...")
        else:
            break

    # 读取视频流中的一帧
    focus_measure = 0
    while focus_measure < 50:
        ret, frame = cap.read()
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        focus_measure = laplacian.var()
        print(focus_measure)
    if not ret:
        print("Failed to read frame")
        return None
    # 将OpenCV图像转换为PIL图像
    frame = Image.fromarray(frame)
    cap.release()
    return frame

i=0
while(1):
    i=i+1
    #start_time = time.time()
    save_path = 'OIP-C.jpg'
    # 保存图像到指定路径
    image=Image.open(save_path)
    #esp32_cam_url = "http://192.168.4.1:81/stream"
    #image = capture_picture_from_esp32_cam(esp32_cam_url)
    #image = cv2.imread('1.jpg')
    #image = Image.fromarray(image)
    infor_image=image_captioning(image)
    #end_time = time.time()
    #print("耗时: {:.2f}秒".format(end_time - start_time))
    pulse = 70
    location = 'classroom'
    prompt=llama_1(infor_image,pulse,location)
    #end_time = time.time()
    #print("耗时: {:.2f}秒".format(end_time - start_time))
    content = prompt['content']

    # 将字符串分割成单独的句子
    sentences = content.split('\n')

    # 提取第1到第2个句子
    extracted_sentences = sentences[len(sentences)//2:len(sentences)//2+1]  # 索引从1开始，因为第0个元素是标题

    # 打印结果
    for sentence in extracted_sentences:
        print(sentence)
    prompt_accurate=llama(sentence,pulse,location)
    #end_time = time.time()
    #print("耗时: {:.2f}秒".format(end_time - start_time))
    musicgen(prompt_accurate['content'],'times_'+str(i))
    #end_time = time.time()
    #print("耗时: {:.2f}秒".format(end_time - start_time))
