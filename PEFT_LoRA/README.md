# 基于PEFT库使用LORA进行FineTune
## 核心代码
```python
    # 配置lora config
    model_lora_config = LoraConfig(
        r = 32, 
        lora_alpha = 32, # scaling = lora_alpha / r 一般来说，lora_alpha的参数初始化为与r相同，即scale=1
        init_lora_weights = "gaussian", # 参数初始化方式
        target_modules = ["linear1", "linear2"], # 对应层添加lora层
        lora_dropout = 0.1
    )

    # 两种方式生成对应的lora模型，调用后会更改原始的模型
    new_model1 = get_peft_model(origin_model, model_lora_config)
    new_model2 = LoraModel(origin_model, model_lora_config, "default")
```
## Run
```
python main.py
```
