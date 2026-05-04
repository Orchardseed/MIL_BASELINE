import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.general_utils import add_epoch_info_log, early_stop, init_epoch_info_log, set_global_seed
from utils.loop_utils import train_loop, val_loop
from utils.model_utils import get_criterion, get_model_from_yaml, get_optimizer, get_scheduler, model_select, save_last_model, save_log
from utils.process_utils import get_process_pipeline
from utils.wsi_utils import WSI_Dataset


def process_FOURIERMIL(args):
    train_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, 'train')
    val_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, 'val')
    test_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, 'test')
    process_pipeline = get_process_pipeline(val_dataset, test_dataset)
    args.General.process_pipeline = process_pipeline

    generator = torch.Generator()
    generator.manual_seed(args.General.seed)
    set_global_seed(args.General.seed)
    num_workers = args.General.num_workers
    if args.Dataset.balanced_sampler.use:
        sampler = train_dataset.get_balanced_sampler(replacement=args.Dataset.balanced_sampler.replacement)
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=num_workers, generator=generator, sampler=sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    print('DataLoader Ready!')

    device = torch.device(f'cuda:{args.General.device}')
    mil_model = get_model_from_yaml(args).to(device)
    print('Model Ready!')

    optimizer, base_lr = get_optimizer(args, mil_model)
    scheduler, warmup_scheduler = get_scheduler(args, optimizer, base_lr)
    criterion = get_criterion(args.Model.criterion)
    warmup_epoch = args.Model.scheduler.warmup

    epoch_info_log = init_epoch_info_log()
    best_model_metric = args.General.best_model_metric
    reverse = best_model_metric == 'val_loss'
    best_val_metric = 9999 if reverse else 0
    best_epoch = 1
    print('Start Process!')
    print('Using Process Pipeline:', process_pipeline)
    for epoch in tqdm(range(args.General.num_epochs), colour='GREEN'):
        now_scheduler = warmup_scheduler if epoch + 1 <= warmup_epoch else scheduler
        train_loss, cost_time = train_loop(device, mil_model, train_dataloader, criterion, optimizer, now_scheduler)
        if process_pipeline == 'Train_Val_Test':
            val_loss, val_metrics = val_loop(device, args.General.num_classes, mil_model, val_dataloader, criterion)
            test_loss, test_metrics = val_loop(device, args.General.num_classes, mil_model, test_dataloader, criterion)
        elif process_pipeline == 'Train_Val':
            val_loss, val_metrics = val_loop(device, args.General.num_classes, mil_model, val_dataloader, criterion)
            test_loss, test_metrics = None, None
        else:
            val_loss, val_metrics, test_loss, test_metrics = None, None, None, None
            if epoch + 1 == args.General.num_epochs:
                test_loss, test_metrics = val_loop(device, args.General.num_classes, mil_model, test_dataloader, criterion)

        print('----------------INFO----------------\n')
        print(f'EPOCH:{epoch+1},  Train_Loss:{train_loss},  Val_Loss:{val_loss},  Test_Loss:{test_loss},  Cost_Time:{cost_time}\n')
        print(f'Val_Metrics:  {val_metrics}\n')
        print(f'Test_Metrics:  {test_metrics}\n')
        add_epoch_info_log(epoch_info_log, epoch, train_loss, val_loss, test_loss, val_metrics, test_metrics)
        best_val_metric, best_epoch = model_select(reverse, args, mil_model.state_dict(), val_metrics, best_model_metric, best_val_metric, epoch, best_epoch)
        if early_stop(args, epoch_info_log, process_pipeline, epoch, mil_model.state_dict(), best_epoch):
            break
        if epoch + 1 == args.General.num_epochs:
            save_last_model(args, mil_model.state_dict(), epoch + 1)
            save_log(args, epoch_info_log, best_epoch, process_pipeline)

