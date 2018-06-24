function[train_data,train_label,test_data,test_label]=devide_data_random(data,class,p)
    [m,~]=size(data);
    pos=randperm(m);
    train_data=data(pos(1:int32(m*p)),:);
    train_label=class(pos(1:int32(m*p)),:);
    test_data=data(pos(int32(m*p)+1:m),:);
    test_label=class(pos(int32(m*p)+1:m),:);
end