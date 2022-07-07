import os 
import shutil

title_finetune_dir = 'temp/tmp_data/lhq_output/title_finetune'
save_dir = 'project/best_model/title'
os.makedirs(save_dir, exist_ok=True)

fold_list = ['fold0', 'fold3', 'fold5', 'order']
for fold in fold_list:
    ckpt_dir_path = os.path.join(title_finetune_dir, fold, 'seed0')
    files = os.listdir(ckpt_dir_path)
    best_loss = 10000
    for file in files:
        filename, filesuffix = os.path.splitext(file)
        if filesuffix == '.pth':
            min_loss = float(filename.split('_')[4])
            if min_loss < best_loss:
                best_loss = min_loss
                best_file = file

    print(f'best_loss:{best_loss}')
    print(f'best_file:{best_file}')
    best_file_path = os.path.join(ckpt_dir_path, best_file)
    save_file_path = os.path.join(save_dir, fold+'.pth')
    print(f'best_file_path:{best_file_path}')
    print(f'save_file_path:{save_file_path}')
    shutil.copyfile(best_file_path, save_file_path)