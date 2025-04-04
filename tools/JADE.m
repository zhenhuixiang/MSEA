function [bestP,bestFitness] = JADE(Dim, Max_NFEs,FUN,minerror,ghx)
    addpath('./CauchyFunction');
    
    time_begin=tic;
    %  disp('JADE global search');
    n = Dim; NP = 100; flag_er=0;
    lu = [min(ghx); max(ghx)];
    LowerBound = lu(1, :);
    UpperBound = lu(2, :);

    G=1;
    uCR=0.5;
    uF=0.5;
    p0=0.05;
    top=round(p0*NP);
    A=[];
    t=1;
    c=0.1;
    
    UB=UpperBound;
    LB=LowerBound;
    
    UB=repmat((UB),NP,1);
    LB=repmat((LB),NP,1);
    
    P=(UB-LB).*rand(NP,Dim)+LB; % Randomly generate initial population individuals
    
    fitnessP=FUN(P);            % Calculate the fitness value of individuals in the population
    
    NFEs=NP;                    % Record the number of fitness function calls
    [fitnessBestP,indexBestP]=min(fitnessP);
    bestP=P(indexBestP,:);
    recRMSE(1:NP)=fitnessP;
    
    %% loop
    while NFEs<Max_NFEs
        fitnessBestP_old = fitnessBestP;
        
        Scr=[];                     % The crossover probability of individuals that initially successfully participate in the mutation is an empty set
        Sf=[];                      % The scaling factor of individuals that initially successfully participate in the mutation is an empty set
        n0=1;                       % Record the number of elements in Scr and Sf
        
        % Sort by individual fitness values ​​to get the best top individuals
        [~,indexSortP]=sort(fitnessP);
        
        for i=1:NP
            
            % Calculate the crossover probability and scaling factor for each individual in generation G
            CR(i)=normrnd(uCR,0.1); % Normal distribution
            F(i)=cauchyrnd(uF,0.1); % Cauchy distribution

            if(CR(i)>1)
               CR(i)=1;
            elseif CR(i)<0
               CR(i)=0; 
            end
            while (F(i)<=0)
                F(i)=cauchyrnd(uF,0.1);
            end
            if (F(i)>1)
                F(i)=1;
            end
            
        end
        
        %%Mutation operation + crossover operation
        for i=1:NP  
                                                                                                                                                                                                                                                                                                                                                                                  
            %%Mutation operation
            for j=1:top
               bestTopP(j,:)=P(indexSortP(j),:); 
            end

            %Randomly select one from the top individuals as Xpbest
            k0=randperm(top,1);
            Xpbest=bestTopP(k0,:);

            %Randomly select P1 from the current population P
            k1=randi([1,NP]);
            P1=P(k1,:);
            while (k1==i)
                k1=randi([1,NP]);
                P1=P(k1,:); 
            end

            %Randomly select P2 from P∪A
            PandA=[P;A];
            [num,~]=size(PandA);
            k2=randi([1,num]);
            P2=PandA(k2,:);
            while (k2==i||k2==k1)
                k2=randi([1,num]);
                P2=PandA(k2,:); 
            end
            V(i,:)=P(i,:)+F(i).*(Xpbest-P(i,:))+F(i).*(P1-P2);   
       
            %%crossover operation
            jrand=randi([1,Dim]); 
            for j=1:Dim
                k3=rand;
                if(k3<=CR(i)||j==jrand)
                    U(i,j)=V(i,j);
                else
                    U(i,j)=P(i,j);
                end
            end
            
        end
        
        
        %% Boundary Processing
        for i=1:NP
           for j=1:Dim
              if (U(i,j)>UB(i,j)||U(i,j)<LB(i,j))
                 U(i,j)=(UB(i,j)-LB(i,j))*rand+LB(i,j); 
              end
           end
        end
        
        fitnessU=FUN(U);
        NFEs=NFEs+NP;
        
        for i=1:NP
        %% Select an action

            if(fitnessU(i)<fitnessP(i))
                A(t,:)=P(i,:);
                P(i,:)=U(i,:);
                fitnessP(i)=fitnessU(i);
                Scr(n0)=CR(i);
                Sf(n0)=F(i);
                t=t+1;
                n0=n0+1;
                if(fitnessU(i)<fitnessBestP)
                   fitnessBestP=fitnessU(i);
                   bestP=U(i,:);
                end
            end
            
            recRMSE(NFEs)=fitnessP(i);
        end

        %Determine whether the size of the archived population A is within NP. If it is greater, randomly remove individuals to keep its size within NP.
        [tA,~]=size(A);
        if tA>NP
            nRem=tA-NP;
            k4=randperm(tA,nRem);
            A(k4,:)=[]; 
            [tA,~]=size(A);
            t=tA+1;
        end

        %Adaptive parameters, update uCR and uF
        [~,ab]=size(Scr);
        if ab~=0
            newSf=(sum(Sf.^2))/(sum(Sf));
            uCR=(1-c)*uCR+c.*mean(Scr);
            uF=(1-c)*uF+c.*newSf;
        end
        %% Determine the termination condition
        error=abs(fitnessBestP_old-fitnessBestP); 
        if error <= minerror   % minerror termination condition
            flag_er=flag_er+1;
        else
            flag_er=0;
        end
        if flag_er >=10
            break;
        end
        
        % disp(['Iteration ' num2str(G) '   Best Cost = ' num2str(fitnessBestP) '   NFE=' num2str(NFEs)]);
        G=G+1;
        
    end
    bestFitness=fitnessBestP;
    endNFEs = NFEs;
    time_cost=toc(time_begin);
end