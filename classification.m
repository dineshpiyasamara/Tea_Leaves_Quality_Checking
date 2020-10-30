outputFolder = fullfile('tea_leaf');
rootFolder = fullfile(outputFolder,'class');
categories = {'best','below_best','poor'};
imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});

imds = splitEachLabel(imds,minSetCount,'randomize');
countEachLabel(imds);

best = find(imds.Labels == 'best',1);
normal = find(imds.Labels == 'below_best',1);
poor = find(imds.Labels == 'poor',1);

net = resnet50();

net.Layers(1);
net.Layers(end);

numel(net.Layers(end).ClassNames);
[trainingSet,testSet] = splitEachLabel(imds,0.1,'randomize');

imageSize = net.Layers(1).InputSize;

augmentedTrainingSet = augmentedImageDatastore(imageSize,...
    trainingSet,'ColorPreprocessing','gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize,...
    testSet,'ColorPreprocessing','gray2rgb');

w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);

featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer,...
    'MiniBatchSize',32,'OutputAs','columns');

trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures,trainingLabels,...
    'Learner','Linear','Coding','onevsall','ObservationsIn','columns');

testFeatures = activations(net, augmentedTestSet, featureLayer,...
    'MiniBatchSize',32,'OutputAs','columns');

predictLabels = predict(classifier,testFeatures,'ObservationsIn','columns');

testLabels = testSet.Labels;
confMat = confusionmat(testLabels,predictLabels);
confmat = bsxfun(@rdivide,confMat,sum(confMat,2));

acc = mean(diag(confMat));
fprintf('\naccuracy is : %0.2f\n',acc);

newImage = imread(fullfile('12.jpg'));
ds = augmentedImageDatastore(imageSize,newImage,'ColorPreprocessing','gray2rgb');
imageFeatures = activations(net,ds,featureLayer,...
    'MiniBatchSize',32,'OutputAs','columns');

label = predict(classifier,imageFeatures,'ObservationsIn','columns');
fprintf('\nImage belongs to --> %s\n\n',label)







