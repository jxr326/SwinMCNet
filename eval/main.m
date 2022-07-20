clc;
clear all;

predpath     = 'C:\Users\Desktop\VT723_maps\Ours\';
maskpath     = 'C:\Users\Desktop\VT723\GT\';


names = dir(fullfile(maskpath,'*.png'));
disp(names);
names = {names.name};
wfm          = 0; mae    = 0; sm     = 0; fm     = 0; prec   = 0; rec    = 0; em     = 0;
score1       = 0; score2 = 0; score3 = 0; score4 = 0; score5 = 0; score6 = 0; score7 = 0;

results      = cell(numel(names), 6);
ALLPRECISION = zeros(numel(names), 256);
ALLRECALL    = zeros(numel(names), 256);
a_fth      = zeros(numel(names), 256);
a_th       = zeros(numel(names), 256);
file_num     = false(numel(names), 1);
for k = 1:numel(names)
    name        = names{1,k};
    results{k, 1} = name;
    file_num(k)   = true;
    
    gtpath = [maskpath name];
    gt     = imread(gtpath);

    fgpath = [predpath strrep(name, '.jpg', '.png')]; 
    fg     = imread(fgpath);  
        
    if length(size(fg)) == 3, fg = fg(:,:,1); end
    if length(size(gt)) == 3, gt = gt(:,:,1); end
    fg = imresize(fg, size(gt));
    fg = mat2gray(fg);
    gt = mat2gray(gt);
    if max(fg(:)) == 0 || max(gt(:)) == 0, continue; end
    
    gt(gt>=0.5) = 1; gt(gt<0.5) = 0; gt = logical(gt);
    score1                   = MAE(fg, gt);
    [score2, score3, score4] = Fmeasure(fg, gt, size(gt));
    score5                   = wFmeasure(fg, gt);
    score6                   = Smeasure(fg, gt);
    score7                   = Emeasure(fg, gt);
    mae                      = mae  + score1;
    prec                     = prec + score2;
    rec                      = rec  + score3;
    fm                       = fm   + score4;
    wfm                      = wfm  + score5;
    sm                       = sm   + score6;
    em                       = em   + score7;
    results{k, 2}            = score1;
    results{k, 3}            = score4;
    results{k, 4}            = score5;
    results{k, 5}            = score6;
    results{k, 6}            = score7;
    [precision, recall]      = PRCurve(fg*255, gt);
    ALLPRECISION(k, :)       = precision;
    ALLRECALL(k, :)          = recall;
 
    [all_f_th all_th]        = Fm_th(fg, gt, size(gt));
    a_fth(k, :)              = all_f_th;
    a_th(k, :)               = all_th;
    
end
m_fth    = mean(a_fth, 1);
prec     = mean(ALLPRECISION(file_num,:), 1);
rec      = mean(ALLRECALL(file_num,:), 1);
maxF     = max(1.3*prec.*rec./(0.3*prec+rec+eps));
file_num = double(file_num);
fm       = fm  / sum(file_num);
mae      = mae / sum(file_num);
wfm      = wfm / sum(file_num);
sm       = sm  / sum(file_num);
em       = em  / sum(file_num);

fprintf('%6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f\n', fm, maxF, wfm, mae, em, sm)

save_path = 'C:\Users\Desktop\VT723_eval\PRcurve\Ours\';
fprintf('The save path is %s\n', save_path);
if ~exist(save_path, 'dir'), mkdir(save_path); end
save([save_path 'results.mat'], 'results');
save([save_path 'prec.mat'], 'prec');
save([save_path 'rec.mat'], 'rec');


save_path = 'C:\Users\Desktop\VT723_eval\Fmeasure\';
fprintf('The save path is %s\n', save_path);
if ~exist(save_path, 'dir'), mkdir(save_path); end
save([save_path 'Ours.mat'], 'm_fth');