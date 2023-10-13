close;clear;clc;
%% Section 1: importData

[FileName,PathName,~] = uigetfile('*.*','Please select point cloud data');
[filepath,name,ext] = fileparts(FileName) ;
tf=strcmpi(ext,'.pcd');
if tf==1
    ptCloud=pcread([PathName,FileName]);
    pcData=ptCloud.Location;
else
    pcData=importdata([PathName,FileName]);

%transform to pointcloud formmat
end

%% Section 2: Parameter Calculation
tic
[m,n]=size(pcData(:,1:3));
pcnormal=zeros(m,n);%normal vectors
s=zeros(m,1);%curvature
% k=input('please input the vaule K (recommended:20-40):');
k=40;
[idx,d] = knnsearch(pcData(:,1:3),pcData(:,1:3),"K",k);
parfor ii=1:m
    Qi=pcData(idx(ii,:),1:3);
    [coeff,score,latent] = pca(Qi);
    s(ii)=latent(3)/sum(latent).*100;
    pcnormal(ii,:)=coeff(:,3);
end
pcData(:,4:6)=pcnormal;
pcData(:,7)=s;
timeFeature=toc;
% Section 3: Flip the normal vector if it is not pointing towards the sensor.

sensorCenter = [1,1,1]; 
pcnormal_flip=pcnormal;
for k = 1 : size(pcnormal,1)
   p1 = sensorCenter;
   p2 = pcnormal(k,1:3);
   angle=acos(dot(p1,p2)/(norm(p1)*norm(p2)))*180/pi;
   if angle > 90
       pcnormal_flip(k,1:3) = -pcnormal(k,1:3);
   end
end

pcData(:,4:6)=pcnormal_flip;

%% Section 4: Discard the edges
tic
cdfvalue=input('please input the vaule cumulative probability (recommended:0.8-0.9):');%%
r1=quantile(s,cdfvalue);
ii=1;
while (ii==1)
    ql_pcData=pcData((pcData(:,7)<=r1),:); 
    figure;
    ql_pcData1=ql_pcData;
    ql_pcData1(1,7)=10;
    pcshow(ql_pcData1(:,1:3),ql_pcData1(:,7)) 
    grid on;
    xlabel(gca,'X (m)','fontname','Times New Roman','fontsize',16 );
    ylabel(gca,'Y (m)','fontname','Times New Roman','fontsize',16 );
    zlabel(gca,'Z (m)','fontname','Times New Roman','fontsize',16 );
    set(gcf,'Color','w');
    set(gca,'Color','w');
    set(gca,'XColor',[0 0 0]);
    set(gca,'YColor',[0 0 0]);
    set(gca,'ZColor',[0 0 0]);
    axis equal;
    pause;
    answer = questdlg('Does meet the result?', ...
	    'Yes','No');
    switch answer
        case 'Yes'
            ii = 0;
        case 'No'
            prompt = {'Please enter cdf value:'};
            dlgtitle = 'Input';
            dims = [1 35];
            answer = inputdlg(prompt,dlgtitle,dims);
            cdfvalue = str2num(answer{1});  %#ok<*ST2NM>
            r1 = quantile(s,cdfvalue);
            ii = 1;
    end
end
timeedge=toc;
%% Section 5: Categorize randomly selected points based on FCM with PSO
% 1.PSO initialize
K_C=input('please inpur the number of discontinuity set according to the color represented by normal:'); 
dowmsampleratio=input('please inpur the ratio of downsmple:'); 
jj=1;
ith=1;
iter_max = 1000;%maximum number of iterations
fit_iter = zeros(iter_max,1);
while jj==1
tic;
[~,dim] = size(pcData(:,4:6));
tol = 10e-6;

% pso parameter
pop_num = 1000;
c1 = 1.5;
c2 = 1.5;
w = 0.9;      %inertia coefficient

pos_max = max(pcData(:,4:6));
pos_min = min(pcData(:,4:6));

V_max = 0.05*(pos_max-pos_min);
V_min =  -0.05*(pos_max-pos_min);


% the boundary of velocity and position
for c = 1:K_C
    x_max(1,(c-1)*dim+1:c*dim) = pos_max;
    x_min(1,(c-1)*dim+1:c*dim) = pos_min;
    v_max(1,(c-1)*dim+1:c*dim) = V_max;
    v_min(1,(c-1)*dim+1:c*dim) = V_min;
end
x_pop = zeros(pop_num,K_C*dim);          
v_pop = zeros(pop_num,K_C*dim);          
for i = 1:pop_num
        x_pop(i,1:K_C*dim) = rand(1,K_C*dim).*(x_max-x_min)+ x_min.*ones(1,K_C*dim);%Initialize the position of each particle，equivalent to K_C pair normal vector
                                                                                    
        v_pop(i,1:K_C*dim) = rand(1,K_C*dim).*(v_max-v_min)+ v_min.*ones(1,K_C*dim);%Initialize the velocity of each particle
end           
x_pop_initial=x_pop;
v_pop_initial=v_pop;
%% 2.calculate intitial fitness
  
ptCloud_ql=pointCloud(ql_pcData(:,1:3));
ptCloudout=pcdownsample(ptCloud_ql,'random',dowmsampleratio);
[~,b]=ismember(ptCloudout.Location,ql_pcData(:,1:3),'rows');
dataset=ql_pcData(b,:);
[pop_num,~] = size(x_pop);
    
fitness = zeros(pop_num,1);

% Write each set of cluster centers
	for i = 1:pop_num
	    [fitness(i),~] = assignment(x_pop(i,:),dataset(:,4:6),K_C);
	end


%% 3.finding cluter center by PSO and clustering dataset by FCM algorithm to get learning sample

[~,index] = sort(fitness);                  
gbest = x_pop(index(1),:);  			 	% xgbest represents the position where the particle exhibits the lowest fitness within the entire group
gbest_fitness = fitness(index(1));        	% Minimum population fitness
pbest = x_pop;              				% xpbest refers to the position at which a particle has its lowest fitness
pbest_fitness = fitness;    				% Minimum individual fitness


fit_iter(1,ith) = gbest_fitness; 
iter = 2;
flag = 1; % Precision control, accuracy to meet the requirements set to 0
while(iter <= iter_max && flag == 1)    
    % Update the velocity and position
    for i = 1:pop_num
        % Update the velocity
        v_pop(i,:) = w*v_pop(i,:) + c1*rand*(pbest(i,:)-x_pop(i,:)) + c2*rand*(gbest-x_pop(i,:));
        
        for j = 1:K_C*dim                  % Velocity boundary processing
            if(v_pop(i,j)> v_max(1,j))
                v_pop(i,j) =  v_max(1,j);   
            end
            if(v_pop(i,j)  < v_min(1,j))
                v_pop(i,j) =  v_min(1,j);
            end
        end
        
        %Update the position
        x_pop(i,:) = x_pop(i,:) + 0.5 * v_pop(i,:);
       
        for j = 1:K_C*dim                  		%position boundary processing
            if(x_pop(i,j)> x_max(1,j))
                x_pop(i,j) = x_max(1,j);   		
            end
            if(x_pop(i,j)  < x_min(1,j))
                x_pop(i,j) = x_min(1,j) ;     	
            end
        end
    end %
    
    % The fitness of the updated particle is recalculated, pbest,pbest_fitness,gbest,and gbest_fitness are updated
    fitness = Fitness(x_pop,dataset(:,4:6),K_C);
    for i = 1:pop_num
        
        if (fitness(i) < pbest_fitness(i))
            pbest(i,:) = x_pop(i,:);  
            pbest_fitness(i) = fitness(i);   
   
            if (pbest_fitness(i) < gbest_fitness )
                gbest = pbest(i,:);    
                gbest_fitness = pbest_fitness(i);
            end 
        end       
    end   
    fit_iter(iter,ith) = gbest_fitness;
  
    sumC = 0;
    % 
    if(  iter > 3 )
        for co = 1:3
            sumC = sumC + abs(fit_iter(iter+1-co,ith) - fit_iter(iter-co,ith));
        end
        if(sumC < tol)
            flag = 0;
            cooo = iter;
        end
    end
    
    iter = iter + 1;
end 
[~,res] = assignment(gbest,dataset(:,4:6),K_C);%% categorize the selected points
toc;
time(ith)=toc;
figure(1);
hold on;
plot(fit_iter(1:cooo,ith));
xlabel("iteration");
ylabel('Fitness');
title(['Fuzzy c-means && PSO ', ',','Minimum Fitness:',num2str(fit_iter(cooo))]);
hold off;

if time(ith)>=50
    jj=0;
else
    jj=1;
    ith=ith+1;
end
end
timeSample=sum(time);
%% Section 6: Categorize the entire point cloud using CNN trained by learning samples

tic
groupNum=K_C;
pcLearn=cell(1,groupNum);
for i=1:K_C
    pcLearn{i}=dataset(res(i).index,1:7);
    pcLearn{i}(:,9)=i;
end

% represent the location of sample in point cloud
figure
pcshow(ptCloud);
hold on;
grid on;
set(gca,'fontname','Times New Roman','fontsize',14);
xlabel(gca,'X (m)','fontname','Times New Roman','fontsize',16 );
ylabel(gca,'Y (m)','fontname','Times New Roman','fontsize',16 );
zlabel(gca,'Z (m)','fontname','Times New Roman','fontsize',16 );
set(gcf,'Color','w');
set(gca,'Color','w');
set(gca,'XColor',[0 0 0]);
set(gca,'YColor',[0 0 0]);
set(gca,'ZColor',[0 0 0]);
axis equal;
color=rand(groupNum,3);
for i=1:groupNum
    if isempty(pcLearn{i})
    else
        scatter3(pcLearn{i}(:,1),pcLearn{i}(:,2),pcLearn{i}(:,3),40,color(i,1:3),'filled');
        hold on;
    end
end

%
Tem_pclearn=cat(1,pcLearn{:});
[a,b]=ismember(Tem_pclearn(:,1:3),pcData(:,1:3),'rows');
pclearn=zeros(size(Tem_pclearn,1),size(Tem_pclearn,2));
pclearn(:,1:7)=pcData(b,1:7);
pclearn(:,8)=b;
pclearn(:,9)=Tem_pclearn(:,9);


% train the network use cnn

[m1,n1]=size(pclearn(:,4:6));
Xtrain=double(reshape(pclearn(:,4:6)',1,n1,1,m1));
ytrain= pclearn(:,9);
ytrain=categorical(ytrain); % 函数包要求标签类型是categorical

layers = [
    imageInputLayer([1 3 1],"Name","data")
    convolution2dLayer([1 3],96,"Name","conv1","BiasLearnRateFactor",2,"Padding",'same')
    reluLayer("Name","relu1")
    crossChannelNormalizationLayer(5,"Name","norm1","K",1)
%     maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    groupedConvolution2dLayer([1 3],128,2,"Name","conv2","BiasLearnRateFactor",2,"Padding",'same')
    reluLayer("Name","relu2")
    crossChannelNormalizationLayer(5,"Name","norm2","K",1)
%     maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])
    groupedConvolution2dLayer([1 3],192,2,"Name","conv3","BiasLearnRateFactor",2,"Padding",'same')
    reluLayer("Name","relu3")
    groupedConvolution2dLayer([1 3],192,2,"Name","conv4","BiasLearnRateFactor",2,"Padding",'same')
    reluLayer("Name","relu4")
    groupedConvolution2dLayer([1 3],128,2,"Name","conv5","BiasLearnRateFactor",2,"Padding",'same')
    reluLayer("Name","relu5")
%     maxPooling2dLayer([3 3],"Name","pool5","Stride",[2 2])
    fullyConnectedLayer(4096,"Name","fc6","BiasLearnRateFactor",2)
    reluLayer("Name","relu6")
    dropoutLayer(0.5,"Name","drop6")
    fullyConnectedLayer(4096,"Name","fc7","BiasLearnRateFactor",2)
    reluLayer("Name","relu7")
    dropoutLayer(0.5,"Name","drop7")
    fullyConnectedLayer(groupNum,"Name","fc8","BiasLearnRateFactor",2)
    softmaxLayer("Name","prob")
    classificationLayer("Name","output")];
options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate',0.0001,...
    'MaxEpochs',200,...
    'MiniBatchSize',200, ...
    'GradientThreshold',10, ...
    'Verbose',false, ...
    'Plots','training-progress');
[net,info] = trainNetwork(Xtrain,ytrain,layers,options); % 网络训练
%
timeCNNtraining=toc;
tic
[m2,n2]=size(pcData(:,4:6));    
pcData_input=double(reshape(pcData(:,4:6)',1,n2,1,m2));
Pred = double(classify(net,pcData_input));
pcData(:,8)=Pred;
timeCNNpred=toc;
%% Section 7: shows the results of grouping with one color per discontinuity set
color=rand(groupNum,3);
for ii=1:groupNum
    figure;
    pcshow(pcData(Pred==ii,1:3),pcData(Pred==ii,4:6))
%     pcshow(pcData(Pred==ii,1:3),color(ii,1:3))
    grid on;
    set(gca,'fontname','Times New Roman','fontsize',14);
    xlabel(gca,'X (m)','fontname','Times New Roman','fontsize',16 );
    ylabel(gca,'Y (m)','fontname','Times New Roman','fontsize',16 );
    zlabel(gca,'Z (m)','fontname','Times New Roman','fontsize',16 );
    set(gcf,'Color','w');
    set(gca,'Color','w');
    set(gca,'XColor',[0 0 0]);
    set(gca,'YColor',[0 0 0]);
    set(gca,'ZColor',[0 0 0]);
    axis equal;
end

figure;
for ii=1:K_C   
    pcshow(pcData(Pred==ii,1:3),color(ii,1:3))
    hold on
end
grid on;
set(gca,'fontname','Times New Roman','fontsize',14);
xlabel(gca,'X (m)','fontname','Times New Roman','fontsize',16 );
ylabel(gca,'Y (m)','fontname','Times New Roman','fontsize',16 );
zlabel(gca,'Z (m)','fontname','Times New Roman','fontsize',16 );
set(gcf,'Color','w');
set(gca,'Color','w');
set(gca,'XColor',[0 0 0]);
set(gca,'YColor',[0 0 0]);
set(gca,'ZColor',[0 0 0]);
axis equal;

%% Section 8: HDBSCAN is used to segment discontinuity set to obtain individual discontinuities
% section 8.1
J_pcData=cell(K_C,1);% cell array，each cell stores each discontinuity set
for ii=1:K_C
    J_pcData{ii}=pcData(Pred==ii,1:8);
end
save J_pcData.mat J_pcData;
% section 8.2 
% The hdbscan program for segment discontinuity set runs in python. See hdbscan.py for the code
% When executing hdbscan in python, import the result with the following command

% section 8.3
[path] = uigetdir ('*.*','Select the path where the hdbscan results are saved in python');
for ii=1:K_C
    filename=strcat(path,'\cluster_labels',num2str(ii),'.txt');
J_pcData{ii}(:,9)=importdata(filename);
end
DisTh=200;
for ii=1:K_C
    DelCategory=[];
    numindDis=max(J_pcData{ii}(:,9));
    for jj=0:numindDis
        j=J_pcData{ii}(J_pcData{ii}(:,9)==jj,:);
        if size(j,1)<DisTh
           J_pcData{ii}(J_pcData{ii}(:,9)==jj,9)=-1;
           DelCategory=[DelCategory;jj];
        end        
    end
    for mm=size(DelCategory,1):-1:1
        a=find(J_pcData{ii}(:,9)>DelCategory(mm));
        J_pcData{ii}(a,9)=J_pcData{ii}(a,9)-1;
    end
end
for ii=1:K_C    
    J_pcData{ii}(:,9)=J_pcData{ii}(:,9)+1;
end

%% Section 9: Calculate the orientation
Orient=cell(K_C,1);
for ii=1:K_C
    for jj=1:max(J_pcData{ii}(:,9))
    j=J_pcData{ii}(J_pcData{ii}(:,9)==jj,:);
    pc=pca(j(:,1:3));
    normal=pc(:,3)';
    [dip,dipdirection] = orientation(normal);
    Orient{ii}(jj,1)=dip;
    Orient{ii}(jj,2)=dipdirection;
    end
end
orient=Orient{1};
for ii=2:K_C
orient=cat(1,orient,Orient{ii});
end




