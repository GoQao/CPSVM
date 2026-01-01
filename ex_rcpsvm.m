clear;clc;

for fi=1:45  
    re1=[];
    name=num2str(fi);
    filename1= strcat('./dataset/awa/',name,'.mat');
    gridsave = strcat('./re/awa/awa_',name,'_grid','.csv');
    S=load(filename1);
    data_before=S.x;
    data_before1=mapminmax(data_before',0,1);
    data=data_before1';
    [N,M]=size(data);
    data=[data ones(N,1)];
    label=S.y;
    pi=S.p;
    pi=[pi ones(2,1)];
    p_data=data(1:100,:);n_data=data(101:end,:);


    best_acc=-1;best_gmean=-1;best_precision=-1;best_recall=-1;best_f1=-1;best_spec=-1;best_auc=-1;best_acc_time=-1;
    rand('seed',1)
    indices_p=crossvalind('Kfold',100,5);
    indices_n=crossvalind('Kfold',100,5);

    grid_data=[];
    for lambda=1
        for ic=1:7      %1:7
            c=10^(ic-4);
            for id=1:17   %1:17    
                d=2^(id-9);
                for igamma1=1:7  %1:7
                    g1=10^(igamma1-4);
                    for igamma2=1:7       %1:7
                        g2=10^(igamma2-4);
                        for yta=[0.01 0.1 0.5 1 2 5 10]
                            for k=1:5 
                                tic;
                                TP=0;FP=0;TN=0;FN=0;
                                validation_p = (indices_p == k);
                                validation_n = (indices_n == k);
                                validation=[validation_p;validation_n];
                                train_p=~validation_p;
                                train_n=~validation_n;
                                train_label=label(~validation);
                                A1=p_data(train_p,:);A2=n_data(train_n,:);
                                train_data=[A1;A2];

                                B1=pi(1,:);B2=pi(2,:);
                                validation_data=data(validation,:);
                                validation_label=label(validation);
                                [len,~]=size(validation_data);
                                [j,~] = size(train_data);
                                
                                fprintf('\n***************************** Dataset: %d K: %d  ******************************\n',fi,k);
                                fprintf('\n******** lambda:%4.4e | c: %4.4e | d: %4.4e | g1: %4.4e | g2: %4.4e | yta:%4.4e ********\n',lambda,c,d,g1,g2,yta);
                                bta = (1-exp(-yta))^(-1);
                                v = -ones(j,1);
                                for iis = 1:10
                                    C = (bta*yta)*c * (-v);
                                    model = train_ex_cpi(A1,A2,B1,B2,C,d,lambda,'rbf',g1,g2);
                                    w=model.alpha;
                                    wp=model.beta;
                                    Lh = Lhinge(A1,A2,B1,B2,w,wp,g1,g2);
                                    Z=Lh.Z;
                                    v = -exp(-yta*Z);
                                end
                               
                                label_predict1 = kernel(validation_data,train_data,'rbf',g1,1)*w;
                                label_predict(label_predict1>=0)=1;
                                label_predict(label_predict1<0)=-1;
        
                                for i=1:len
                                    if validation_label(i)==1 && label_predict(i)==1
                                        TP=TP+1;
                                    elseif validation_label(i)==1 && label_predict(i)==-1
                                        FN=FN+1;
                                    elseif validation_label(i)==-1 && label_predict(i)==1
                                        FP=FP+1;
                                    else
                                        TN=TN+1;
                                    end
                                end
                                tt=toc;

                                accuracy(k)=(TP+TN)/(TP+TN+FP+FN);
                                precision(k)=TP/(TP+FP);
                                recall(k)=TP/(TP+FN);
                                F1_measure(k)=(2*TP)/(2*TP+FN+FP);
                                spec(k)=TN/(TN+FP);
                                gmean(k)=(recall(k)*spec(k))^0.5;
                                auc(k)=AUC(validation_label,label_predict1);
                                time(k)=tt;
                            end
                            %------------ acc ---------------
                            inter_acc=mean(accuracy);   
                            inter_astd=std(accuracy);
                            %--------- precision-------------
                            inter_precision=mean(precision);   
                            inter_pstd=std(precision);
                            %----------- recall--------------
                            inter_recall=mean(recall);   
                            inter_rstd=std(recall);
                            %----------- F1_measure ---------
                            inter_F1_measure=mean(F1_measure);   
                            inter_fstd=std(F1_measure);
                            %------------- spec --------------
                            inter_spec=mean(spec);    
                            inter_sstd=std(spec);
                            %------------- gmean --------------
                            inter_gmean=mean(gmean);   
                            inter_gstd=std(gmean);
                            %------------- auc ----------------
                            inter_auc=mean(auc);   
                            inter_aucstd=std(auc);
                            %------------- time ---------------- 
                            inter_time=mean(time);   
                            inter_timestd=std(time);

        
                            grid_data=[grid_data;fi lambda c d g1 g2 yta ...
                                inter_acc inter_astd inter_precision inter_pstd inter_recall inter_rstd...
                                inter_F1_measure inter_fstd inter_spec inter_sstd inter_gmean inter_gstd...
                                inter_auc inter_aucstd inter_time inter_timestd];
        
                            if best_acc<inter_acc
                                best_acc_lambda=lambda;
                                best_acc_c=c;
                                best_acc_d=d;
                                best_acc_g1=g1;
                                best_acc_g2=g2;
                                best_acc_yta=yta;
                                best_acc=inter_acc;
                                best_astd=inter_astd;
                                best_acc_time=inter_time;
                                best_acc_timestd=inter_timestd;
                            end
                            if best_precision<inter_precision                              
                                best_precision=inter_precision;
                                best_pstd=inter_pstd;
                            end
                            if best_recall<inter_recall 
                                best_recall=inter_recall;
                                best_rstd=inter_rstd;
                            end
                            if best_f1<inter_F1_measure
                                best_f1=inter_F1_measure;
                                best_fstd=inter_fstd;
                            end
                            if best_spec<inter_spec
                                best_spec=inter_spec;
                                best_sstd=inter_sstd;
                            end
                            if best_gmean<inter_gmean
                                best_gmean=inter_gmean;
                                best_gstd=inter_gstd;
                            end
                            if best_auc<inter_auc
                                best_auc=inter_auc;
                                best_aucstd=inter_aucstd;
                            end
                        end
                    end
                end
            end
        end
    

        re1=[re1;best_acc_lambda best_acc_c best_acc_d best_acc_g1 best_acc_g2  best_acc_yta best_acc best_astd ...
            best_f1 best_fstd...
            best_gmean best_gstd ...
            best_auc best_aucstd...
            best_precision best_pstd...
            best_recall best_rstd ...
            best_spec best_sstd ...
            best_acc_time best_acc_timestd];
        resu=re1';
        fname = sprintf('./re/awa/rcpsvm_awa_%d_lambda_%d.csv',fi,lambda);
        csvwrite(fname,resu);
    
    
    
        paraNames = {'dataset','lambda','c','d','g1','g2','yta',...
            'acc','acc_std','pre','pre_std','recall','recall_std','fscore','fscore_std','spec','spec_std','gmean','gmean_std','auc','auc_std','time','time_std'};
        s = grid_data;
        T_grid = table(s(:,1),s(:,2),s(:,3),s(:,4),s(:,5), s(:,6),s(:,7), ...
            s(:,8),s(:,9),s(:,10),s(:,11),s(:,12),s(:,13),s(:,14),s(:,15),s(:,16),s(:,17),s(:,18),s(:,19),s(:,20),s(:,21),s(:,22),s(:,23),'VariableNames',paraNames);
        writetable(T_grid,gridsave);
    end
end