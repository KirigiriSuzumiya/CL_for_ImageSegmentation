# Experience TODO list

## Specific Module for Image Segementation

### Benchmark

> [!NOTE]
>
> structure: Scenarios - stream - experience - dataset 

> [!IMPORTANT]
>
> pending
>
> waiting for experience scenarios design   

- [x]  Dataloader
  - [x] use torch dataloader loading files
  - [x] convert torch dataset to avalanche dataset
- [x] experience
  - [x] Convert dataset to CLDatasetExperience
- [x] stream
  - [x] train
  - [x] test
  - [x] Eval
- [x] scenarios
  - [x] (deprecated) default scenarios 
    - [ ] ~~all scenarios is compiled for image classification~~
    - [ ] ~~decide to Overload class~~
  - [x] CL scenarios
    - [x] build from stream
      - [x] compile experiences
    - [x] DataAttribute
      - [x] task label needed for scenarios 
        - [x] edit dataloader policy



### Training



- [x] strategy

  - [x] naive
    - [x] reload naive policy
      - [x] edit minibatch module
      - [x] reload mb_x
      - [x] reload mb_y
      - [x] reload mb_task_label
      - [x] relaod _unpack_minibatch
    - [x] unit test
      - [x] Pass
  - [x] ewc
  - [x] SI
  - [x] LFL
  - [x] GEM

- [x] settings

  - [x] model

    - [x] unet
      - [x] n_channels = 3
      - [x] n_classes =4 

  - [x] optimizer

    - [x] RMSprop
      - [x] lr = 0.00001
      - [x] weight_decay = 0.00000001
      - [x] momentum = 0.999

  - [x] criterion

    - [x] CrossEntropyLoss()

  - [x] Batch_size = 8

  - [x] epoch = 5

  - [x] device

    - [x] Run on cpu for code testing
      - [x] Pass
    - [x] move to cuda
      - [x] cuda:0

    

  

### Evaluation

- [ ] Normal plugin

  - [x] loss_metrics
  - [x] timing_metrics
  - [x] cpu_usage_metrics
  - [ ] forgetting_metrics
  - [ ] (deprecated) accuracy_metrics
    - [x] normal plugin is for image classification
    - [ ] TODO

- [ ] Specific plugin

  - [x] using `strategy.eval()` 




### Report 

- [ ] data storage
  - [x] sending to wandb
    - [x] dev project: `avalanche`
    - [x] prd project: `UNet_CL`
- [ ] metrics
  - [ ] create report