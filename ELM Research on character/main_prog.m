clear;clc;close all;
delete Result_analysis.xlsx;
Lang={'English_num_MNIST','Bangla_num_NITRKL','Odia_num_IIITBBS','Bangla_num_ISIKOL','Odia_num_ISIKOL'};
h_node=[4000];
h_rand={'relu','ortho'};
h_acti={'Relu','LeakyRelu','Gaussian'};
% for ix=1:size(h_actii,2)
%     h_acti={h_actii{ix}};
    h_dev=[80];
    run=1;
    for ilt=1:size(Lang,2)
        full_data=load(strcat(Lang{ilt},'\Data_file.mat'));
        full_data.cls=full(ind2vec((full_data.cls+1)'))';
        clear result;
        kt=1;
        for i=1:size(h_acti,2)
            for j=1:size(h_rand,2)
                for k=1:size(h_node,2)
                    for l=1:size(h_dev,2)
                        for tt=1:run
                            [train_data,train_label,test_data,test_label]=devide_data_random(full_data.data,full_data.cls,h_dev(l)./100);
                            [train_acc(tt),test_acc(tt),~,~]=SLFN_ELM(train_data,train_label,test_data,test_label,h_node(k),h_rand{j},h_acti{i});
                        end
                        result{kt,1}=h_acti{i};
                        result{kt,2}=h_rand{j};
                        result{kt,3}=h_node(k);
                        result{kt,4}=strcat(num2str(h_dev(l)),'-',num2str(100-h_dev(l)));
                        result{kt,5}=mean(train_acc);
                        result{kt,6}=var(train_acc);
                        result{kt,7}=mean(test_acc);
                        result{kt,8}=var(test_acc);
                        kt=kt+1;
                    end
                end
            end
        end
        f={'Activation function','Rand Intialization','Number of Node','Training Testing Ratio','Training Accuracy','Train_var','Testing Accuracy','Test_var'};
        result=[f;result];
        xlswrite(strcat('Result_analysis_final_20_june_mult','.xlsx'),result,strcat(Lang{ilt}),'A1')
    end