GT_PATH = '/home/selfdriving/zhaotiny/SegNet/cityscape/labelsIdx/' % replace with the abosolute address of the GT_PATH
PRE_PATH = '/home/selfdriving/zhaotiny/SegNet/cityscape/city_all2/' % replace with the abosolute address of the PRED_PATH
%fid = fopen('./read_mapping/valIdx.txt'); %% resting file
fid = fopen('read_mapping/cityscape/cityValIdx.txt')
img_name = fgetl(fid);
% number of labels = number of classes plus one for the background
nclasses = 5;
num = nclasses +1; 
confcounts = zeros(num);
count=0;
countImg = 1;
while ischar(img_name)
%     disp(tline)
    
    gt_file = [GT_PATH  img_name];
    pre_file = [PRE_PATH img_name];
    gtimg = imread(gt_file);
    gtim = double(gtimg);
    preimg = imread(pre_file);
    preim = double(preimg);
    
    % Check validity of results image
    maxlabel = max(preim(:));
    if (maxlabel>nclasses)
        error('Results image ''%s'' has out of range value %d (the value should be <= %d)',imname,maxlabel,VOCopts.nclasses);
    end

    szgtim = size(gtim); szpreim = size(preim);
    if any(szgtim~=szpreim)
        error('Results image ''%s'' is the wrong size, was %d x %d, should be %d x %d.',imname,szresim(1),szresim(2),szgtim(1),szgtim(2));
    end
    
    %pixel locations to include in computation
    locs = gtim<255;
%     if (numel(find(preim == 3)) > 0)
%         img_name
%     end
    
    % joint histogram
    sumim = 1+gtim+preim*num; 
    hs = histc(sumim(locs),1:num*num); 
    count = count + numel(find(locs));
    confcounts(:) = confcounts(:) + hs(:);
    
    countImg = countImg + 1;
%     tline = fgetl(fid);
%     if countImg > 2000
%         break;
%     end
    img_name = fgetl(fid);
end
fclose(fid);

% confusion matrix - first index is true label, second is inferred label
%conf = zeros(num);
conf = 100*confcounts./repmat(1E-20+sum(confcounts,2),[1 size(confcounts,2)]);
rawcounts = confcounts;

% Percentage correct labels measure is no longer being used.  Uncomment if
% you wish to see it anyway
%overall_acc = 100*sum(diag(confcounts)) / sum(confcounts(:));
%fprintf('Percentage of pixels correctly labelled overall: %6.3f%%\n',overall_acc);

accuracies = zeros(nclasses,1);
fprintf('Accuracy for each class (intersection/union measure)\n');
for j=1:num
   
   gtj=sum(confcounts(j,:));
   resj=sum(confcounts(:,j));
   gtjresj=confcounts(j,j);
   % The accuracy is: true positive / (true positive + false positive + false negative) 
   % which is equivalent to the following percentage:
   accuracies(j)=100*gtjresj/(gtj+resj-gtjresj);   
   
%    clname = 'background';
%    if (j>1), clname = VOCopts.classes{j-1};end;
%    fprintf('  %14s: %6.3f%%\n',clname,accuracies(j));
end
accuracies = accuracies(1:end);
avacc = mean(accuracies);
fprintf('-------------------------\n');
fprintf('Average accuracy: %6.3f%%\n',avacc);
save('ds.mat', 'conf')
