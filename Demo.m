%% Setup
%clear variables;
net = alexnet;
inputSize = net.Layers(1).InputSize;
categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face', 'not'};

%% Load Images
img1 = imread('G:\My Drive\Image Rec Project\Data\Manually_Annotated\1\73eda9bf85e7f63215f6c2b247b8b70be2cd48a5d10bebfecf636176.jpg');
img1 = uint8(imresize(img1, [inputSize(1) inputSize(2)]));
img2 = imread('G:\My Drive\Image Rec Project\Data\Manually_Annotated\1\90633de5a85bed0365b000e6f91e11a2064e41fddeab3809bbfe0ea1.jpg');
img2 = uint8(imresize(img2, [inputSize(1) inputSize(2)]));
img3 = imread('G:\My Drive\Image Rec Project\Data\Manually_Annotated\1\bb40207f33f4edae9a01125e4fc9d74df4a097696f87dec57c9d995a.jpg');
img3 = uint8(imresize(img3, [inputSize(1) inputSize(2)]));
img4 = imread('G:\My Drive\Image Rec Project\Data\Manually_Annotated\8\82bb346162f6d2583b986292dd9b56b6496757cd116f423bfa76b253.jpg');
img4 = uint8(imresize(img4, [inputSize(1) inputSize(2)]));
scoreClassifications = zeros(11, 1);

for category =1:11
    curCat = categories(category);
    filename = strcat('oneEmotion\', curCat{1}, '.mat');
    load(filename);
    img = img4;
    
    curCat = categories(category);
    [Ypred, scores] = classify(netTransfer, img);
    if(Ypred(1) == "not" && scores(1,1) >=.5) || (Ypred(1) ~= "not" && scores(1,1) <=.5)
        scoreClassifications(category, :) = scores(:,2);
        
    else
        scoreClassifications(category, :) = scores(:,1);
    end
    classfications(category, :) = Ypred;
    
end
maxScore = max(scoreClassifications);
index = scoreClassifications == maxScore;
finalCategory = classfications(index);
%%
figure(1);
%subplot(2,1,1);
imshow(img);
figTittle = char(finalCategory);
title(figTittle);
% figure(2);
categories = categories(1:11);
uitable('Data', scoreClassifications', 'ColumnName', categories, 'Position',[400 70 700 50]);


