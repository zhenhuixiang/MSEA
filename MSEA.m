%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%  MSEA: 
%  This code is part of the program that produces the results in the following paper:
%  Huixiang Zhen, Offline evolutionary optimization with problem-driven model pool design and weighted model selection indicator, under review, 2025.
%  This matlab code was written by Huixiang Zhen, School of Computer Science, China University of Geoscience. 
%  Date: 4/4/2025
%  Written by Huixiang Zhen, zhenhuixiang@cug.edu.cn.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;
close;
addpath(genpath(pwd));
warning off all

% Experiment setting
testset = "fname";
funcArr={'S_ELLIPSOID'; 'S_ROSENBROCK'; 'S_ACKLEY'; 'S_GRIEWANK'; 'S_RASTRIGIN'; 'CEC05_F10'; 'CEC05_F16'; 'CEC05_F19';}; 
dimsArr = [10 30];                 % dimsArr = [10 30 50 100];
o = length(funcArr);
d = size(dimsArr,2);
runs = 2;
out = [];

% Parameters setting
p_t = 2;                           % The number of times the test data and training data are divided (default 1)
p_top = 0.2;                       % test data selected from p_top data (default 0.2)
p_r = 0.5;                         % EA search region based on p_top data (default 0.5)

% Experiment
for j = 1:o 
    outFUN = [];
    for i = 1:d 
        dims = dimsArr(i);
        result = [];
        time = [];
        pos = [];

        for r = 1:runs
            %% 1.Initialization
            % 1.1 Initialize offline data
            fname = cell2mat(funcArr(j));  
            FUN=@(x) feval(fname,x); 
            [Xmin, Xmax] = variable_domain(fname);
            [fname ' D=' num2str(dims) ' R=' num2str(r)]
            LB = repmat((Xmin),1,dims);
            UB = repmat((Xmax),1,dims,1);
            initial_sample_size = 11*dims;
            sam=repmat(LB,initial_sample_size,1)+(repmat(UB,initial_sample_size,1)-repmat(LB,initial_sample_size,1)).*rand(initial_sample_size,dims);
            fitness = FUN(sam);
            hx=sam; hf=fitness';
            DATA = [hx, hf];                            % generated offline Data
            t1 = clock;                                 % time start
            
            % 1.2 Initialize elite data number
            [sort_hf,id]=sort(hf); 
            sort_hx = hx(id,:);
            nd = length(hf);
            num_elite_data = floor(nd*p_r);             % elite data

            % 1.3 Initialize model pool number and selection metrics
            model_num = 4;
            test_error = zeros(1,model_num);
            co_Evaluate = zeros(1,model_num);
            
            %% 2.1 Calculate test_error [C1]
            test_subset_num = p_t;
            for j2 = 1:test_subset_num
                % Divided dataset to train and test dataset 
                topnum = floor(0.1*initial_sample_size); 
                index = randperm(floor(p_top*initial_sample_size),topnum);
                test_hf = sort_hf(index); test_hx = sort_hx(index,:);
                train_hf = sort_hf;  train_hx = sort_hx;
                train_hf(index) = [];  train_hx(index,:) = [];

                % Build models based on train data and compute test_error and sort_error
                % Exact interpolation RBF [model 1]
                i = 1;
                ghxd=real(sqrt(train_hx.^2*ones(size(train_hx'))+ones(size(train_hx))*(train_hx').^2-2*train_hx*(train_hx')));
                spr=max(max(ghxd))/(dims*nd)^(1/dims);
                net=newrbe(train_hx',train_hf',spr);
                RBF_FUN=@(x) sim(net,x');
                RBF_FUN_Arr{i} = RBF_FUN;
                error = abs(test_hf - RBF_FUN(test_hx)'); % test error
                test_error(i) = test_error(i) + sum(error);
                
                % Overall trend RBF [model 2]
                i = 2;
                ghxd=real(sqrt(train_hx.^2*ones(size(train_hx'))+ones(size(train_hx))*(train_hx').^2-2*train_hx*(train_hx')));
                spr=max(max(ghxd));
                net=newrbe(train_hx',train_hf',spr);
                RBF_FUN=@(x) sim(net,x');
                RBF_FUN_Arr{i} = RBF_FUN;
                error = abs(test_hf - RBF_FUN(test_hx)'); % test error
                test_error(i) = test_error(i) + sum(error);
                
                % Ensemble of RBF-L and RBF-G [model 3]
                i = 3;
                RBF_FUN_Arr_ensemble = RBF_FUN_Arr([1,2]);
                ensemble = @(x)ensemble_RBF_FUN(x,RBF_FUN_Arr_ensemble);
            	RBF_FUN_Arr{3} = ensemble;
                error = abs(test_hf - ensemble(test_hx)');
                test_error(i) = test_error(i) + sum(error);
                
                % Ensemble model of many RBFs [model 4]
                i = 4;
                [ensemble_RBF_time,ensemble_model] = ensemble_RBF(dims,[train_hx, train_hf],UB,LB);
                ensemble_model2 = @(x) ensemble_model(x)';
                RBF_FUN_Arr{4} = ensemble_model2;
                error = abs(test_hf - ensemble_model(test_hx));
                test_error(i) = test_error(i) + sum(error); 
            end
            test_error = test_error/test_subset_num; %%%%%%%%%%%%%%%%% test_error [C1]
            
            %% 2.2 Calculate co_Evaluate C2
            % Predict by build models with all data [search], where RBF network use all data
            predict_pos = [];
            if true
                elite_data = sort_hx(1:num_elite_data,:); 
                % Search predicted optimal based on [model 1]
                i = 1;
                ghxd=real(sqrt(hx.^2*ones(size(hx'))+ones(size(hx))*(hx').^2-2*hx*(hx')));
                spr=max(max(ghxd))/(dims*nd)^(1/dims);
                net=newrbe(hx',hf',spr);
                RBF_FUN=['RBF_FUN',num2str(i)];
                eval([RBF_FUN,'=@(x) sim(net,x'');']);
                eval(['RBF_FUN=',RBF_FUN,';']);
                maxgen = 200000+500*dims;
                minerror = 1e-20;
                RBF_FUN_Arr{i} = RBF_FUN;
                [best_pos,bestever] = JADE(dims, maxgen, RBF_FUN, minerror, elite_data);
                predict_pos(i,:) = best_pos;
                
                % Search predicted optimal based on [model 2]
                i = 2;
                ghxd=real(sqrt(hx.^2*ones(size(hx'))+ones(size(hx))*(hx').^2-2*hx*(hx')));
                spr=max(max(ghxd))/(dims*nd)^(1/dims)*2^12;
                net=newrbe(hx',hf',spr);
                RBF_FUN=['RBF_FUN',num2str(i)];
                eval([RBF_FUN,'=@(x) sim(net,x'');']);
                eval(['RBF_FUN=',RBF_FUN,';']);
                maxgen = 200000+500*dims;
                minerror = 1e-20;
                RBF_FUN_Arr{i} = RBF_FUN;
                [best_pos,bestever] = JADE(dims, maxgen, RBF_FUN, minerror, elite_data);
                predict_pos(i,:) = best_pos;
    
                % Search predicted optimal based on [model 3]
                i = 3;
                maxgen = 200000+500*dims; 
                minerror = 1e-20;
                RBF_FUN_Arr_ensemble = RBF_FUN_Arr([1,2]);
                ensemble = @(x)ensemble_RBF_FUN(x,RBF_FUN_Arr_ensemble);
                RBF_FUN_Arr{i} = ensemble;
                [best_pos, bestever] = JADE(dims, maxgen, ensemble, minerror, elite_data);
                predict_pos(i,:) = best_pos;
                
                % Search predicted optimal based on [model 4]
                i = 4;
                [ensemble_RBF_time,ensemble_model] = ensemble_RBF(dims,[hx,hf],UB,LB);
                maxgen = 200000+500*dims; 
                minerror = 1e-20;
                ensemble_model2 = @(x) ensemble_model(x)';
                RBF_FUN_Arr{i} = ensemble_model2;
                [best_pos,bestever] = JADE(dims, maxgen, ensemble_model2, minerror, elite_data);
                predict_pos(i,:) = best_pos;
            end
            % Calculate mutual evaluation by predicted position C2
            for m = 1:model_num
                RBF_FUN = RBF_FUN_Arr{m};
                co_E_temp(m,:) = RBF_FUN(predict_pos);
                co_E_temp(m,m) = 0;
            end
            co_Evaluate = sum(co_E_temp,1); %[co_E]
            
            %% 2.3 Weighted Indicator
            C1 = mapminmax(test_error, 0, 1);  % Normalize to [0,1]
            C2 = mapminmax(co_Evaluate, 0, 1); % Normalize to [0,1]
            C = C1 + 0.05 * C2; 
            
            %% 2.4 Select and caldulate result
            [~, model_selected] =  min(C);
            pos = predict_pos(model_selected,:);
            result(r) = FUN(pos)                % Real fitness evaluation
            totaltime = etime(clock,t1);
            time(r) = totaltime;
        end

        %% 3.Statistic result
        best_result = min(result, [], 2);
        worst_result = max(result, [], 2);
        mean_result = mean(result, 2);
        median_result = median(result, 2);
        std_result    = std(result, 0, 2);
        mean_time = mean(time);
        
        %% 4.1 output function stats result
        out_ONE_FUN = [best_result,worst_result,mean_result,median_result,std_result;];
        outFUN = [outFUN; best_result(end),worst_result(end),mean_result(end),median_result(end),std_result(end);];
        out = [out; best_result(end),worst_result(end),mean_result(end),median_result(end),std_result(end);];
        save(strcat('result/', mfilename, '_', fname, ' D', num2str(dims) ,' runs=',num2str(runs)),'out_ONE_FUN','result','mean_time','time','pos');
    end

    % 4.2 Output function with all dimension
    save(strcat('result/', mfilename, '_', fname, ' runs=',num2str(runs)),'outFUN');
end

% 4.3 Output all
save(strcat('result/', mfilename, '_', 'all_', ' runs=',num2str(runs)),'out');