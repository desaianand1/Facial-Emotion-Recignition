function [finalCategory, scores, index] = simpleClassifer(img)
%SIMPLECLASSIFER runs the given img through all the one emotion classifers
%and returns the category that had the highest confidence
N = size(img, 4);
scoreClassifications = zeros(11, N);
%classfications = cell(11, N);
categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face', 'not'};

for category =1:11
    curCat = categories(category);
    filename = strcat('..\oneEmotion\', curCat{1}, '.mat');
    load(filename);
    [Ypred, scores] = classify(netTransfer, img);
    if(Ypred(1) == "not" && scores(1,1) >=.5) || (Ypred(1) ~= "not" && scores(1,1) <=.5)
        classfications(category, :) = Ypred;
        scoreClassifications(category, :) = scores(:,2);
        
    else
        classfications(category, :) = Ypred;
        scoreClassifications(category, :) = scores(:,1);
    end

end

classfications = categorical(classfications);
maxScore = max(scoreClassifications);
index = scoreClassifications == maxScore;
finalCategory = classfications(index);
scores = scoreClassifications;
end

