function [x_new_1,y_new_1,x_new_2,y_new_2] = localization_refinement(strr,x1,y1,x2,y2,color)
%LOCALIZATION_REFINEMENT 此处显示有关此函数的摘要
%   此处显示详细说明
original_image=imread(strr);
h = size(original_image, 1);
w = size(original_image, 2);
c_x = round((x1 + x2) / 2);
c_y = round((y1 + y2) / 2);
dia_x = abs(x1-x2);
dia_y = abs(y1-y2);
original_image = imread(strr);
% 新框大小
up = max(1, c_x-dia_x);
down = min(h, c_x+dia_x);
left = max(1, c_y-dia_y);
right = min(w, c_y+dia_y);
rgb_image = original_image(up:down,left:right,:); %裁取出的新框
% figure(100);
% imshow(rgb_image);
% title('resize之前的图像');
for i=1:3
    temp(:,:,i) =imresize(rgb_image(:,:,i),[250,250],'bilinear');
end
rgb_image=temp;
% figure(1)
% imshow(rgb_image);
% title('原采集图像');
double_rgb=double(rgb_image);

%基于归一化RGB彩色空间R分量的分割，一定程度上可以去除光照影响
sum_rgb=double_rgb(:,:,1)+double_rgb(:,:,2)+double_rgb(:,:,3);
r_image=double_rgb(:,:,1)./sum_rgb;
g_image=double_rgb(:,:,2)./sum_rgb;
b_image=double_rgb(:,:,3)./sum_rgb;
rbgn=cat(3,r_image,g_image,b_image);
% figure(2)
% imshow(rbgn);
% title('归一化后的RGB图像');

%对于红色颜色的判定与分割,使用文献中的方法，增强红色区域
if color == "red"
    BW=max(0,min(r_image-b_image,r_image-g_image)); % 红色区域增强
end
if color == "blue"
    BW=max(0,min(b_image-r_image,b_image-g_image));  % 蓝色区域增强
end
if color == "yellow"
    BW=max(0,min(r_image-b_image,g_image-b_image));  % 黄色区域增强
end
% figure(12)
% imshow(BW)

%使用最大间类方差法（大津法）对R分量进行阈值分割
thresh=graythresh(BW);
BW=im2bw(BW,thresh);
% figure(3)
% imshow(BW);
% title('阈值分割后图像');

%基于形态学的处理，去除毛刺等噪声，而且可以减少小面积的连通区域数目
%要求拍摄的照片中交通标志所占面积比例不能太小，否则分割不出来
BW=imopen(BW,ones(3,3));  %经过多次试验，使用闭运算更合适，确定结构元素的尺寸为3*3
% figure(4)
% imshow(BW);
% title('形态学开运算处理后图像');

%轮廓提取与区域填充
BW=imfill(BW,'holes');  %区域填充
% figure(5)
% imshow(BW);      %BW是区域特征
% title('填充之后');

se=strel('square',3);
erode_image=imerode(BW,se);
bound_image=BW-erode_image;
% figure(6)
% imshow(bound_image);   %bound_image是边界特征
% title('边界提取后的图像');
%连通区域（对于一个像素点边界形成的区域）
[L,NUM]=bwlabel(BW,8);
%disp(['图中共有' num2str(NUM) '个连通分量'])

%若有多个区域，只保留面积最大的前3个区域
switch NUM
    case 0
        errordlg(' 没有检测到可能的圆形禁令交通标志');
        %break;
    case 1
        leave_BW{1}=L;
         [t(1),area_ration(1),r(1),a(1),b(1)]= certificate_circle(leave_BW{1});  %圆形检测
        %figure
        %imshow(leave_BW{1}),title('可能的交通标志区域');
        %break;
    case 2
        leave_BW{1}=(L==1);
        leave_BW{2}=(L==2);
        %f_l=figure('name','分割得到的2个主要区域');
        %subplot(1,2,1),imshow(leave_BW{1});
        %subplot(1,2,2),imshow(leave_BW{2});
        [t(1),area_ration(1),r(1),a(1),b(1)]= certificate_circle(leave_BW{1});   %圆形检测
        [t(2),area_ration(2),r(2),a(2),b(2)]= certificate_circle(leave_BW{2});
        %break;
    otherwise
        for k=1:NUM
            [y,x]=find(L==k);  %编号为k的连通区域的坐标，第一个元素是y值，第二个元素是x值
            area(k)=length(x); %标号为k的区域像素值（面积）
        end
        [maxunm,maxindex]=sort(area,'descend'); %index存放最大的3个区域索引
        max_area_index=maxindex(1:3);
        leave_BW{1}=(L==max_area_index(1));
        leave_BW{2}=(L==max_area_index(2));
        leave_BW{3}=(L==max_area_index(3));
        %f_l=figure('name','分割得到的3个主要区域');
        %subplot(1,3,1),imshow(leave_BW{1});
        %subplot(1,3,2),imshow(leave_BW{2});
        %subplot(1,3,3),imshow(leave_BW{3});
        [t(1),area_ration(1),r(1),a(1),b(1)]= certificate_circle(leave_BW{1});    %圆形检测
        [t(2),area_ration(2),r(2),a(2),b(2)]= certificate_circle(leave_BW{2});
        [t(3),area_ration(3),r(3),a(3),b(3)]= certificate_circle(leave_BW{3});
end

for k=1:length(t)
    if t(k)==1
        %分割出交通标志时，要留有一定空白，保证标志的完整性
        row_1=b(k)-r(k)*1.1;
        row_2=b(k)+r(k)*1.1;
        col_1=a(k)-r(k)*1.1;
        col_2=a(k)+r(k)*1.1;
        row_1=round(row_1);
        row_2=round(row_2);
        col_2=round(col_2);
        col_1=round(col_1);
        a1 = row_2;
        b1 = col_1;
        a2 = row_1;
        b2 = col_2;
        l = w / 2;
        r = h / 2;
        x_new_1 = round(2*w/250*a1 + x1 - l);
        y_new_1 = round(2*h/250*b1 + y2 - r);
        x_new_2 = round(2*w/250*a2 + x1 - l);
        y_new_2 = round(2*h/250*b2 + y2 - r);
        if x_new_1 < 1
            x_new_1 = 1;
        end
        if x_new_1 > h
            x_new_1 = h;
        end
        if x_new_2 < 1
            x_new_2 = 1;
        end
        if x_new_2 > h
            x_new_2 = h;
        end
        if y_new_1 < 1
            y_new_1 = 1;
        end
        if y_new_1 > w
            y_new_1 = w;
        end
        if y_new_2 < 1
            y_new_2 = 1;
        end
        if y_new_2 > w
            y_new_2 = w;
        end
    end
end
