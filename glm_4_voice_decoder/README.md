# GLM-4-Voice-Decoder

GLM-4-Voice 是智谱 AI 推出的端到端语音模型。GLM-4-Voice 能够直接理解和生成中英文语音，进行实时语音对话，并且能够根据用户的指令改变语音的情感、语调、语速、方言等属性。

GLM-4-Voice is an end-to-end voice model launched by Zhipu AI. GLM-4-Voice can directly understand and generate Chinese and English speech, engage in real-time voice conversations, and change attributes such as emotion, intonation, speech rate, and dialect based on user instructions.

本仓库是 GLM-4-Voice 的 speech decoder 部分。GLM-4-Voice-Decoder 是基于 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 重新训练的支持流式推理的语音解码器，将离散化的语音 token 转化为连续的语音输出。最少只需要 10 个音频 token 即可开始生成，降低对话延迟。

The repo provides the speech decoder of GLM-4-Voice. GLM-4-Voice-Decoder is a speech decoder supporting streaming inference, retrained based on [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), converting discrete speech tokens into continuous speech output. Generation can start with as few as 10 audio tokens, reducing conversation latency.

更多信息请参考我们的仓库 [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice).

For more information please refer to our repo [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice).