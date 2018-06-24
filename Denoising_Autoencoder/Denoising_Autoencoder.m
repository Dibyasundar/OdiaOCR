clear;clc;close all;

%% Initialisation of POI Libs
% Add Java POI Libs to matlab javapath
if isunix
    javaaddpath('../datasets/poi_library/poi-3.8-20120326.jar');
    javaaddpath('../datasets/poi_library/poi-ooxml-3.8-20120326.jar');
    javaaddpath('../datasets/poi_library/poi-ooxml-schemas-3.8-20120326.jar');
    javaaddpath('../datasets/poi_library/xmlbeans-2.3.0.jar');
    javaaddpath('../datasets/poi_library/dom4j-1.6.1.jar');
    javaaddpath('../datasets/poi_library/stax-api-1.0.1.jar');
end

%%%main code

Lang={'English_num_MNIST','Bangla_num_NITRKL'};%,'Odia_num_IIITBBS','Bangla_num_ISIKOL','Odia_num_ISIKOL'};
h_node={[100,50]};%,[200,50]};
h_dev=[60,80];
iter=[100,400];
run=1;
for ilt=1:size(Lang,2)
    if isunix
        full_data=load(strcat('../datasets/',Lang{ilt},'/Data_file.mat'));
    elseif ispc
        full_data=load(strcat('..\datasets\',Lang{ilt},'\Data_file.mat'));
    end
    full_data.cls=full(ind2vec((full_data.cls+1)'))';
    clear result;
    kt=1;
    for ll=1:size(h_dev,2)
        for k=1:size(h_node,2)
            for kk=1:size(iter,2)
                nod_sz=h_node{1};
                for tt=1:run
                    [train_data,train_label,test_data,test_label]=devide_data_random(full_data.data,full_data.cls,h_dev(ll)./100);
                    sizeDataSample=size(train_data,1);
                    inputSize=size(train_data,2);
                    xTrain = zeros(1,inputSize,1,sizeDataSample);
                    tTrain=train_label;
                    class_size=size(tTrain,2);
                    xOut = zeros(inputSize,sizeDataSample);
                    for i = 1:sizeDataSample
                        o_img=reshape(train_data(i,:),[28,28]);
                        t_img=imnoise(o_img,'gaussian',0,0.02);
                        xTrain(1,:,1,i) = t_img(:); 
                        xOut(:,i)=o_img(:);
                    end
                    rng('default')
                    hiddenSize1 = nod_sz(1);
                    layers=[imageInputLayer([1,inputSize]) fullyConnectedLayer(hiddenSize1,'Name','m1_1') fullyConnectedLayer(inputSize) regressionLayer];
                    opts=trainingOptions('sgdm','MaxEpochs',iter(kk),'InitialLearnRate',0.0001);
                    net1=trainNetwork(xTrain,xOut',layers,opts);
                    feat1=activations(net1,xTrain,'m1_1');

                    xTrain = zeros(1,hiddenSize1,1,sizeDataSample);
                    xOut = zeros(sizeDataSample,hiddenSize1);
                    for i=1:sizeDataSample
                        o_img=feat1(1,1,:,i);
                        t_img=imnoise(o_img,'gaussian',0,0.02);
                        xTrain(1,:,1,i) = t_img; 
                        xOut(i,:)=o_img;
                    end

                    rng('default')
                    hiddenSize2 = nod_sz(2);
                    layers=[imageInputLayer([1,hiddenSize1]) fullyConnectedLayer(hiddenSize2,'Name','m2_1') fullyConnectedLayer(hiddenSize1) regressionLayer];
                    opts=trainingOptions('sgdm','MaxEpochs',iter(kk),'InitialLearnRate',0.0001);
                    net2=trainNetwork(xTrain,xOut,layers,opts);
                    feat2=activations(net2,xTrain,'m2_1');

                    xTrain = zeros(1,hiddenSize2,1,sizeDataSample);%numel(xTrainImages));
                    xOut = zeros(sizeDataSample,1);%numel(xTrainImages));
                    for i=1:sizeDataSample
                        o_img=feat2(1,1,:,i);
                        t=train_label(i,:);
                        t_img=imnoise(o_img,'gaussian',0,0.02);
                        xTrain(1,:,1,i) = t_img; 
                        xOut(i,1)=find(t==1);
                    end
                    for i=1:class_size
                        cls_name{1,i}=char(strcat('c',num2str(i)));
                    end
                    xOut=categorical(xOut,[1:class_size],cls_name);
                    layers=[imageInputLayer([1,hiddenSize2]) fullyConnectedLayer(class_size,'Name','m3_1') softmaxLayer classificationLayer];
                    opts=trainingOptions('sgdm','MaxEpochs',iter(kk),'InitialLearnRate',0.0001);
                    net3=trainNetwork(xTrain,xOut,layers,opts);


                    xTrain = zeros(1,inputSize,1,sizeDataSample);
                    for i = 1:sizeDataSample
                        xTrain(1,:,1,i) = train_data(i,:);
                    end

                    l=fullyConnectedLayer(hiddenSize2);
                    t=net2.Layers(2,1);
                    l.Weights=t.Weights;
                    l.Bias=t.Bias;
                    l2=l;

                    l=fullyConnectedLayer(class_size);
                    t=net3.Layers(2,1);
                    l.Weights=t.Weights;
                    l.Bias=t.Bias;
                    l3=l;

                    f_layer=[imageInputLayer([1,inputSize]) net1.Layers(2,1) l2 l3 net3.Layers(3:end)'];
                    opts=trainingOptions('sgdm','MaxEpochs',iter(kk),'InitialLearnRate',0.0001);
                    final_net=trainNetwork(xTrain,xOut,f_layer,opts);


                    xobt=classify(final_net,xTrain);
                    [~,~,test_obt]=unique(xobt);
                    out=confusionmat(test_obt',vec2ind(train_label'));
                    train_acc(tt)=sum(diag(out))/sum(sum(out));


                    xTest = zeros(1,inputSize,1,size(test_data,1));
                    xTestOut = zeros(size(test_data,1),1);
                    for i = 1:size(test_data,1)
                        xTest(1,:,1,i) = test_data(i,:); 
                    end


                    xobt=classify(final_net,xTest);
                    [~,~,test_obt]=unique(xobt);
                    out=confusionmat(test_obt',vec2ind(test_label'));
                    test_acc(tt)=sum(diag(out))/sum(sum(out));

                end
                result{kt,1}=iter(kk);
                result{kt,2}=strcat(num2str(nod_sz(1)),',',num2str(nod_sz(2)));
                result{kt,3}=strcat(num2str(h_dev(ll)),'-',num2str(100-h_dev(ll)));
                result{kt,4}=mean(train_acc);
                result{kt,5}=var(train_acc);
                result{kt,6}=mean(test_acc);
                result{kt,7}=var(test_acc);
                kt=kt+1;
            end
        end
    end
        f={'Iteration','Number of Node','Training Testing Ratio','Training Accuracy','Train_var','Testing Accuracy','Test_var'};
        result=[f;result];
        if isunix
            xlwrite(strcat('Result_analysis.xls'),result,strcat(Lang{ilt}),'A1');
        elseif ispc
            xlswrite(strcat('Result_analysis.xls'),result,strcat(Lang{ilt}),'A1');
        end
end
