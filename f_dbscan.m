function [ T ] = f_dbscan( A , eps, ppcluster)
% [ T, eps ] = f_dbscan( A , npb, ppcluster)
% Input:
% - A: matriz con las coordenadas de los puntos 
% - eps: radio para b��squeda de vecinos   
% - ppcluster: n m��nimo de puntos por cl��ster 
% Output:
% - T: cl��sters asignados a cada vecino T=zeros(n,1); [n,d]=size(A); 
%    Copyright (C) {2015}  {Adri��n Riquelme Guill, adririquelme@gmail.com}

[n,d]=size(A);
h=waitbar(0,['Cluster analysis in process. ',num2str(n),' points. Please wait']);

minpts=d+1; %minium number of eps-neighbors to consider into a cluster  
T=zeros(n,1);   
maxcluster=1; % 
% 0 sin cl��ster asignado
% 1,2.... cl��ster asignado
% calculamos los puntos dentro del radio de eps
[idx, ~] = rangesearch(A,A,eps);
for i=1:n
    NeighborPts=idx{i};
    % si ha encontrado el m��nimo de puntos, hacer lo siguiente
    % cuidado, el primer ��ndice de idx es el mismo punto
    if length(NeighborPts)>=minpts %el punto es un core point
        % ?el punto tiene cl��ster asignado?
        cv=T(NeighborPts); %cl��ster vecinos
        mincv=min(cv); % 
        mincv2=min(cv((cv>0))); % 
        maxcv=max(cv);% 
        if maxcv==0
            caso=0; % maxcv==0��
        else
            if maxcv==mincv2
                caso=1; % maxcv~=0 && maxcv==mincv2
            else
                caso=2; % maxcv~=0 && maxcv~=mincv2
            end
        end
        switch caso
            case 0
                % ning��n punto tiene c��ster asingado, se lo asignamos
                T(NeighborPts)=maxcluster; % 
                % T(i)=maxcluster;
                maxcluster=maxcluster+1; %
            case 1
                if mincv==0
                    % 
                    T(NeighborPts(cv==0))=mincv2;
                end
                % T(i)=mincv2;
            case 2
                
                T(NeighborPts(cv==0))=mincv2;
                % reagrupamos los puntos que ya tienen cl��ster
                b=cv(cv>mincv2); % cl��sters a reasignar
                [~,n1]=size(b);
                aux=0;
                for j=1:n1
                    if b(j)~=aux
                        T(T==b(j))=mincv2;
                        aux=b(j);
                    end
                end
                % T(i)=mincv2;
        end
    else
        %el punto no tiene suficientes vecinos.
    end
    waitbar(i/n,h);
end
%% homogeneizamos la salida
% si la salida est�� vac��a, es decir que no se encuentra ning��n cluster, no hacemos nada  
if sum(T)==0 
    % no hademos nada, la salida est�� vac��a
    % como todos los puntos tienen valor cero, se eliminar��n despu��s 
else
    % en esta fase cogemos los cl��sters obtenidos y eliminamos los que no
    % superen los N (ppcluster)
    % se ordenan los cl��sters seg��n mayor a menor n? de puntos obtenidos
    T2=T;
    cluster=unique(T2,'sorted');
    cluster=cluster(cluster>0); % eliminamos los cl��sters ru��do 
    [ nclusters,~]=size(cluster);
    % calculamos el n��mero de puntos que pertenecen a cada cluster
    A=zeros(2,nclusters);
    numeroclusters=zeros(1, nclusters);
    for ii=1:nclusters
        numeroclusters(ii)=length(find(T2(:,1)==cluster(ii,1)));
    end
    A(2,:)=cluster; A(1,:)=numeroclusters;   
    % ordeno la matriz seg��n el n��mero de cl��sters encontrados
    [~,IX]=sort(A(1,:),'descend'); A=A(:,IX);
    % buscamos aquellos clusters con m��s de n puntos  
    n=ppcluster;
    I=find(A(1,:)>n);
    J=find(A(1,:)<=n);
    % los cl��sters no significativos le asingamos le valor 0 
    for ii=1:length(J)
        T(T2==A(2,J(ii)))=0;
    end
    % renombramos los cl��sters seg��n importancia 
    for ii=1:length(I)
        T(T2==A(2,I(ii)))=ii;
    end
end
close(h);