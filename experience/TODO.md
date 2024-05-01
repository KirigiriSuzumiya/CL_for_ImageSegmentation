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

- [ ]  Dataloader
  - [x] use torch dataloader loading files
  - [x] convert torch dataset to avalanche dataset
- [ ] experience
  - [x] Convert dataset to CLDatasetExperience
- [x] stream
  - [x] train
  - [x] test
  - [x] Eval
- [ ] scenarios
  - [ ] (deperated) default scenarios 
    - [ ] ~~all scenarios is compiled for image classification~~
    - [ ] ~~decide to Overload class~~
  - [ ] CL scenarios
    - [x] build from stream
      - [x] compile experiences
    - [ ] DataAttribute
      - [x] task label needed for scenarios 
        - [x] edit dataloader policy



### Training



- [ ] stratergy

  - [x] naive
    - [x] reload naive policy
      - [x] edit minibatch module
      - [x] reload mb_x
      - [x] reload mb_y
      - [x] reload mb_task_label
      - [x] relaod _unpack_minibatch
    - [x] unit test
      - [x] Pass
  - [ ] ewc
  - [ ] SI
  - [ ] LFL
  - [ ] GEM

- [ ] settings

  - [x] model

    - [x] unet
      - [x] n_channels = 3
      - [x] n_classes =4 

  - [x] optimizer

    - [x] RMSprop
      - [x] lr = 0.00001
      - [x] weight_decay = 0.00000001
      - [x] momentum = 0.999

  - [ ] criterion

    - [x] CrossEntropyLoss()

  - [x] Batch_size = 8

  - [x] epoch = 5

  - [ ] device

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
  - [ ] (deperated) accuracy_metrics
    - [x] normal plugin is for image classification
    - [ ] TODO

- [ ] Specific plugin

  - [ ] accuracy_metrics for image segementation

  