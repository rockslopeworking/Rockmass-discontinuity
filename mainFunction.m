close;clear;
%% importData
[FileName,PathName,~] = uigetfile('*.*','Please selectpoint cloud data');
[filepath,name,ext] = fileparts(FileName) ;
tf=strcmpi(ext,'.pcd');
if tf==1
    ptCloud=pcread([PathName,FileName]);
    pcData=ptCloud.Location;
else
    pcData=importdata([PathName,FileName]);

%transform to pointcloud formmat
    ptCloud=pointCloud(pcData(:,1:3));
end
%% Parameter Calculation
tic
[m,n]=size(pcData(:,1:3));
pcnormal=zeros(m,n);%normal vectors
s=zeros(m,1);%curvature
k=input('please input the vaule K (recommended:40-60):');
[idx,d] = knnsearch(pcData(:,1:3),pcData(:,1:3),"K",k);

for ii=1:m
    Qi=pcData(idx(ii,:),1:3);
    [coeff,score,latent] = pca(Qi);
    pcnormal(ii,:)=coeff(:,3);
    s(ii)=latent(3)/sum(latent).*100;
end
pcData(:,4:6)=pcnormal;
pcData(:,7)=s;
toc
%%
sensorCenter = [1,1,1]; 
pcnormal_flip=pcnormal;
for k = 1 : size(pcnormal,1)
   p1 = sensorCenter;
   p2 = pcnormal(k,1:3);
   % Flip the normal vector if it is not pointing towards the sensor.
   angle=acos(dot(p1,p2)/(norm(p1)*norm(p2)))*180/pi;
   if angle > 90
       pcnormal_flip(k,1:3) = -pcnormal(k,1:3);
   end
end

pcData(:,4:6)=pcnormal_flip;

%% Discard the edges
cdfvalue=0.85;
r1=quantile(s,cdfvalue);
ii=1;
while (ii==1)
    ql_pcData=pcData((pcData(:,7)<=r1),:); 
    figure;
    pcshow(ql_pcData(:,1:3),ql_pcData(:,4:6)) 
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

%% Seletc sample based on kmeans algorithm with PSO
%% 1.PSO initialize
K_C=input('please inpur the number of discontinuity set according to the color represented by normal:'); 
dowmsampleratio=input('please inpur the ratio of downsmple:'); 
jj=1;
ith=1;
iter_max = 1000;
fit_iter = zeros(iter_max,1);
while jj==1
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
tic;  
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


%% 3.finding cluter center by PSO and clustering dataset by kmearns algorithm to get train sample

[~,index] = sort(fitness);                  
gbest = x_pop(index(1),:);  			 	% xgbest represents the position where the particle exhibits the lowest fitness within the entire group
gbest_fitness = fitness(index(1));        	% Minimum population fitness
pbest = x_pop;              				% xpbest refers to the position at which a particle has its lowest fitness
pbest_fitness = fitness;    				% Minimum individual fitness


fit_iter(1,ith) = gbest_fitness; 
iter = 2;
flag = 1; %Precision control, accuracy to meet the requirements set to 0
while(iter <= iter_max && flag == 1)    
    %Update the velocity and position
    for i = 1:pop_num
        %Update the velocity
        v_pop(i,:) = w*v_pop(i,:) + c1*rand*(pbest(i,:)-x_pop(i,:)) + c2*rand*(gbest-x_pop(i,:));%rand是[0,1]随机数 
        
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
  
    sum = 0;
    % 
    if(  iter > 3 )
        for co = 1:3
            sum = sum + abs(fit_iter(iter+1-co,ith) - fit_iter(iter-co,ith));
        end
        if(sum < tol)
            flag = 0;
            cooo = iter;
        end
    end
    
    iter = iter + 1;
end 
[~,res] = assignment(gbest,dataset(:,4:6),K_C);
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
%% automatically add lable to sample
%% data normalization
tic
pcDataNew=normalize(pcData,'range',[0 1]);
res_foraddsample=res;
pcLearn=cell(1,K_C);
for i=1:K_C
    pcLearn{i}=dataset(res_foraddsample(i).index,:);
    pcLearn{i}(:,9:8+K_C)=zeros(size(pcLearn{i},1),K_C);
    pcLearn{i}(:,8+i)=1;
end
Tem_pclearn=cat(1,pcLearn{:});
[a,b]=ismember(Tem_pclearn(:,1:3),pcData(:,1:3),'rows');
pclearn=zeros(size(Tem_pclearn,1),size(Tem_pclearn,2));
pclearn(:,1:7)=pcDataNew(b,1:7);
pclearn(:,8)=b;
pclearn(:,9:8+K_C)=Tem_pclearn(:,9:8+K_C);

% train the network

net_input=pclearn(:,1:7)';
net_target=pclearn(:,9:8+K_C)';
hiddenLayerSize = 12;
net = patternnet(hiddenLayerSize);

[net,tr] = train(net,net_input,net_target);


% Categorize the entire point cloud

Set_result= round(net(pcDataNew(:,1:7)'))';
toc
%% represent the results of grouping
for ii=1:K_C
    figure;
    pcshow(pcData(Set_result(:,ii)==1,1:3),pcData(Set_result(:,ii)==1,4:6))
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

color=rand(K_C,3); 
figure;
for ii=1:K_C   
    pcshow(pcData(Set_result(:,ii)==1,1:3),color(ii,1:3))
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

%% DBSCAN
J_pcData=cell(K_C,1);
for ii=1:K_C
    J_pcData{ii}=pcData(Set_result(:,ii)==1,:);
end

for ii=1:K_C
    nvecinos=4+1;
    [m,~]=size(J_pcData{ii});
    [~,dist]=knnsearch(J_pcData{ii}(:,1:3),J_pcData{ii}(:,1:3),'k',nvecinos);
    data=dist(:,nvecinos);
    data=unique(data,'sorted');
    eps=mean(data)+2*std(data);
    display([mean(data) mean(data)+2*std(data)]);
    % eps is in the range of the average distance of the 4th nearest neighbor point in each set to average distance plus two standard deviations
d=1;
while d==1
    prompt = {'please input eps:','please input minCluster:'};
    dlgtitle = 'Input';
    dims = [1 50];
    definput = {eps,100};
    answer = inputdlg(prompt,dlgtitle,dims);
    eps  = str2num(answer{1});  %#ok<*ST2NM>
    minCluster  = str2num(answer{2});  %#ok<*ST2NM>

    if isempty(minCluster)  
        minCluster =20;
    end
    J_pcData{ii}(:,8+K_C)=f_dbscan( J_pcData{ii}(:,1:3) , eps, minCluster );
    a=1;
    m=max(J_pcData{ii}(:,8+K_C));
    color=rand(m,3);
    J=J_pcData{ii}(find(J_pcData{ii}(:,8+K_C)~=0),:);%#ok<FNDSB>
    figure;
    for jj=1:m
        j=J_pcData{ii}(J_pcData{ii}(:,8+K_C)==jj,:);
        pcshow(j(:,1:3),color(a,:))
        grid on;
        set(gca,'fontname','Times New Roman','fontsize',14);
        xlabel(gca,'X (mm)','fontname','Times New Roman','fontsize',16 );
        ylabel(gca,'Y (mm)','fontname','Times New Roman','fontsize',16 );
        zlabel(gca,'Z (mm)','fontname','Times New Roman','fontsize',16 );
        view(-115,15);
        axis equal;
        hold on;
        set(gcf,'Color','w');
        set(gca,'Color','w');
        set(gca,'XColor',[0 0 0]);
        set(gca,'YColor',[0 0 0]);
        set(gca,'ZColor',[0 0 0]);
        a=a+1;
    end
    pause
    answer = questdlg('Does the clustering result is satisfactory?');
    switch answer
        case 'Yes'  
            d=0;
            eval(['J',num2str(ii),'=J',';']);
        case 'No'  
            d=1;
    end
end
end
disp('Clustering is complete！');

%% Calculate the orientation
Orient=cell(K_C,1);
for ii=1:K_C
    for jj=1:max(J_pcData{ii}(:,8+K_C))
    j=J_pcData{ii}(J_pcData{ii}(:,8+K_C)==jj,:);
    pc=pca(j(:,1:3));
    normal=pc(:,3)';
    [dd,dip] = orientation(normal);
    Orient{ii}(jj,1)=dd;
    Orient{ii}(jj,2)=dip;
    end
end
orient=Orient{1};
for ii=2:K_C
orient=cat(1,orient,Orient{ii});
end
orient(:,2)=orient(:,2);

