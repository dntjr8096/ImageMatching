
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <Windows.h>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <time.h>

using namespace std;
using namespace cv;

// ī�� ����ü
typedef struct Card {
	CvPoint p1;      // (max_x, max_y)
	CvPoint p2;      // (min_x, min_y)
	int width;
	int height;
	int label;
}Card;

IplImage* g_image = NULL;			//���� �̹���
IplImage* g_gray = NULL;			//gray scale�� �̹���
IplImage* g_binary = NULL;			//edge �̹���
IplImage* g_marked_original = NULL;	//Marking�� �̹���
IplImage** card_image_set = NULL;	//ī�� �̹��� ����
CvMemStorage* g_storage = NULL;		//�����

int g_thresh = 180;      //Edge detecting ���
int card_num;				//ī�� ����
int** map;					//��
int **visit, **temp_visit;	//��õ�� �˰��� ����
int prev_card_num = 0;		//���� �����ӿ����� ī�� ����

Mat hwnd2mat(HWND hwnd);
void on_trackbar(int pos);
CvScalar color_select(int label);
void cardMatch(Card card_set[], int card_num);
double templMatch(Card &card1, Card &card2);
void Initialize_visit_array(int height, int width);
void Initialize_temp_visit_array(int height, int width);
bool To_up(int x, int y, int cardnum);
bool To_down(int x, int y, int cardnum, int height);
bool To_right(int x, int y, int cardnum, int width);
bool To_left(int x, int y, int cardnum);
int find_pair(int height, int width);
void highlight(Card* card_set, int index);

//HWND ������ Mat���� �ٲ��ִ� �Լ�
Mat hwnd2mat(HWND hwnd)
{
	HDC hwindowDC, hwindowCompatibleDC;

	int height, width, srcheight, srcwidth;
	HBITMAP hbwindow;
	Mat src;
	BITMAPINFOHEADER  bi;

	hwindowDC = GetDC(hwnd);
	hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
	SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

	RECT windowsize;    // ��ũ���� height�� weight �������� ����
	GetClientRect(hwnd, &windowsize);

	srcheight = windowsize.bottom;
	srcwidth = windowsize.right;
	height = windowsize.bottom;
	width = windowsize.right;

	//��ũ�� ��ü ����
	src.create(height, width, CV_8UC4);

	//��Ʈ�� ����
	hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
	bi.biSize = sizeof(BITMAPINFOHEADER);
	bi.biWidth = width;
	bi.biHeight = -height;  //�׸��� ������ �Ųٷ�
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 0;
	bi.biXPelsPerMeter = 0;
	bi.biYPelsPerMeter = 0;
	bi.biClrUsed = 0;
	bi.biClrImportant = 0;

	//������ ���� ��ġ�� ��Ʈ�� ����
	SelectObject(hwindowCompatibleDC, hbwindow);
	//������ ��ġ�� ������ ��Ʈ�ʿ� ����
	StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, SRCCOPY);
	GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO *)&bi, DIB_RGB_COLORS);

	//�޸� ����
	DeleteObject(hbwindow);
	DeleteDC(hwindowCompatibleDC);
	ReleaseDC(hwnd, hwindowDC);

	return src;
}

//Ʈ���ٿ� ���� ���� ó���ϴ� �Լ�
void on_trackbar(int pos) {

	//�ʱ�ȭ �۾�
	if (g_storage == NULL) {
		g_gray = cvCreateImage(cvGetSize(g_image), 8, 1);
		g_binary = cvCreateImage(cvGetSize(g_image), 8, 1);
		g_marked_original = cvCreateImage(cvGetSize(g_image), 8, g_image->nChannels);
		g_storage = cvCreateMemStorage(0);
	}
	else {
		cvClearMemStorage(g_storage);
	}

	//Gray Scaling
	cvCvtColor(g_image, g_gray, CV_BGR2GRAY);

	//Binarization
	cvThreshold(g_gray, g_gray, g_thresh, 255, CV_THRESH_BINARY);
	cvCopy(g_gray, g_binary);

	//�ܰ��� ã��
	CvSeq* contours = 0;

	card_num = cvFindContours(   //num���� �ܰ��� ������ ����
		g_gray,               //�Է¿���
		g_storage,            //����� �ܰ����� ����ϱ� ���� �޸� ���丮��
		&contours,            //�ܰ����� ��ǥ���� ����� Sequence
		sizeof(CvContour),
		CV_RETR_EXTERNAL      //���� �ٱ��� �ܰ����� ǥ��
	);

	//�̹����� ���� (�ܰ����� �׸��� ���ؼ�)
	cvZero(g_gray);

	if (contours) {
		//�ܰ����� ã�� ����(contour)�� �̿��Ͽ� �ܰ����� �׸�
		cvDrawContours(
			g_gray,                //�ܰ����� �׷��� ����
			contours,              //�ܰ��� Ʈ���� ��Ʈ���
			cvScalarAll(255),      //�ܺ� �ܰ����� ����
			cvScalarAll(128),      //���� �ܰ����� ����
			100                    //�ܰ����� �׸��� �̵��� ����
		);
	}

	//���ʿ��� ������ ���̱� ���ؼ� ī�� ������ 2�� �̻��̰�, ���� �������� ī�� ������ �ٸ� ��츸 ���� (�ƴ� ��� �׳� return)
	if (prev_card_num != card_num && card_num >= 2) {

		//�ܰ����� ������ŭ ī�� ����
		Card* card_set = new Card[card_num];

		//�ܰ����� ������ �̿��ؼ� ī�� ��ġ ���
		int j = 0;
		for (CvSeq* c = contours; c != NULL; c = c->h_next) {
			for (int i = 0; i < c->total; i++) {
				CvPoint* p = (CvPoint*)cvGetSeqElem(c, i);
				if (p->x >= g_image->width || p->x < 0 || p->y >= g_image->height || p->y < 0)
					return;

				if (i == 0) {
					card_set[j].p1.x = p->x;
					card_set[j].p2.x = p->x;
					card_set[j].p1.y = p->y;
					card_set[j].p2.y = p->y;
				}
				if (p->x > card_set[j].p1.x)
					card_set[j].p1.x = p->x;
				if (p->x < card_set[j].p2.x)
					card_set[j].p2.x = p->x;
				if (p->y > card_set[j].p1.y)
					card_set[j].p1.y = p->y;
				if (p->y < card_set[j].p2.y)
					card_set[j].p2.y = p->y;
			}
			j++;
		}

		//�޸� ����
		cvClearSeq(contours);

		int max_size = 0, max_index = 0;

		//width, height, max_size, max_index ���
		for (int i = 0; i < card_num; i++) {
			card_set[i].width = card_set[i].p1.x - card_set[i].p2.x + 1;
			card_set[i].height = card_set[i].p1.y - card_set[i].p2.y + 1;
			card_set[i].label = 0;
			if (max_size < (card_set[i].width * card_set[i].height)) {
				max_size = card_set[i].width * card_set[i].height;
				max_index = i;
			}
		}

		int* check = new int[card_num];
		int check_num = 0;

		//���ʿ��� ������ ���̱� ���ؼ� �ǹ� ���� edge�� üũ
		for (int i = 0; i < card_num; i++) {
			check[i] = 0;
			for (int j = i + 1; j < card_num; j++) {
				if (card_set[i].p1.x <= card_set[j].p1.x && card_set[i].p1.y <= card_set[j].p1.y  && card_set[i].p2.x >= card_set[j].p2.x && card_set[i].p2.y >= card_set[j].p2.y) {
					check[i] = 1;
					check_num++;
					break;
				}
				else if (card_set[i].width / card_set[i].height >(45 / 50) || card_set[i].width / card_set[i].height < (40 / 55)) {
					check[i] = 1;
					check_num++;
					break;
				}
				else if (card_set[i].width < card_set[max_index].width - 5 || card_set[i].height < card_set[max_index].height - 5) {
					check[i] = 1;
					check_num++;
					break;
				}
			}
		}

		//üũ�� edge�� ����
		card_num -= check_num;
		Card* temp = new Card[card_num];
		j = 0;
		for (int i = 0; i < card_num; i++) {
			while (check[j] != 0)
				j++;
			temp[i].p1 = card_set[j].p1;
			temp[i].p2 = card_set[j].p2;
			temp[i].width = card_set[j].width;
			temp[i].height = card_set[j].height;
			temp[i].label = card_set[j].label;
			j++;
		}

		//�޸� ����
		delete[] check;
		delete[] card_set;

		//�ǹ� �ִ� edge��� ī�� set �籸��
		card_set = temp;

		//���� ī�� ���� ����
		prev_card_num = card_num;

		//ī�� ���� 20���� ����
		//������ edge���� �����ϰ� 2�� �̻� 20�� ������ ���� ���� (�ƴ� ��� ��Ī���� �ʰ� ��� edge�� mark)
		if (card_num <= 20 && card_num >= 2) {

			//ī�� ��Ī
			cardMatch(card_set, card_num);

			int c_w, c_h; // card_width, card_height
			c_w = card_set[0].p1.x - card_set[0].p2.x;
			c_h = card_set[0].p1.y - card_set[0].p2.y;

			if (c_w == 0 || c_h == 0)
				return;

			//map_minx, map_miny, map_maxx, map_maxy ���
			int m_minx, m_maxx, m_miny, m_maxy;
			for (int i = 0; i < card_num; i++) {
				if (i == 0) {
					m_maxx = card_set[i].p1.x;
					m_maxy = card_set[i].p1.y;
					m_minx = card_set[i].p2.x;
					m_miny = card_set[i].p2.y;
				}
				if (card_set[i].p1.x > m_maxx)
					m_maxx = card_set[i].p1.x;
				if (card_set[i].p1.y > m_maxy)
					m_maxy = card_set[i].p1.y;
				if (card_set[i].p2.x < m_minx)
					m_minx = card_set[i].p2.x;
				if (card_set[i].p2.y < m_miny)
					m_miny = card_set[i].p2.y;
			}

			//ī��ũ�� �̿��Ͽ� �� ũ�� ���ϰ� �迭�� ��ȯ
			int rows, cols;
			cols = (m_maxx - m_minx) / c_w;
			rows = (m_maxy - m_miny) / c_h;

			//�� Ȯ���� ���ؼ� 1ĭ�� �� ������ �߰� �Ҵ�

			map = (int**)malloc(sizeof(int*)*(rows + 2));
			for (int i = 0; i < (rows + 2); i++) {
				map[i] = (int*)malloc(sizeof(int)*(cols + 2));
			}
			for (int i = 0; i < rows + 2; i++)
				for (int j = 0; j < cols + 2; j++)
					map[i][j] = -1;

			//��õ�� �˰����� ���� visit, temp_visit �Ҵ�
			visit = (int**)malloc(sizeof(int*)*(rows + 2));
			for (int i = 0; i < (rows + 2); i++) {
				visit[i] = (int*)malloc(sizeof(int)*(cols + 2));
			}
			temp_visit = (int**)malloc(sizeof(int*)*(rows + 2));
			for (int i = 0; i < (rows + 2); i++) {
				temp_visit[i] = (int*)malloc(sizeof(int)*(cols + 2));
			}

			//ī������ �迭�� �ֱ�
			int div_w = (m_maxx - m_minx) / cols;
			int div_h = (m_maxy - m_miny) / rows;
			for (int i = 0; i < card_num; i++) {
				int x = 0, y = 0;
				int c_ix, c_iy; // ī�� x��ǥ ����, ī�� y��ǥ ����
				c_ix = (card_set[i].p1.x + card_set[i].p2.x) / 2;
				c_iy = (card_set[i].p1.y + card_set[i].p2.y) / 2;

				for (int j = 0; j < rows; j++) {
					if (c_iy >= (m_miny + (j)*div_h) && c_iy < (m_miny + (j + 1)*div_h)) {
						y = j;
						break;
					}
				}
				for (int j = 0; j < cols; j++) {
					if (c_ix >= (m_minx + (j)*div_w) && c_ix < (m_minx + (j + 1)*div_w)) {
						x = j;
						break;
					}
				}
				map[y + 1][x + 1] = card_set[i].label;
			}

			cvCopy(g_image, g_marked_original);
			int index = find_pair(rows + 2, cols + 2);

			if (index > 0) {
				//���� �̹������� ī�� ��ġ�� ��ŷ(���̶�����)
				for (int i = 0; i < card_num; i++) {
					if (card_set[i].label == index) {
						highlight(card_set, i);
					}
				}
			}
			else {
				printf("Failed to find pair\n");
			}

			/*  ������ ī��� �̹��� ���� ��� ��������ν� ī�带 �� �ν��ϴ��� Ȯ���ϴ� �κ� (������ ������ ���� ��� �Ǵ� �κ�)

			//ī�� �̹��� 20�� �̸�
			char* name[20] = { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t" };

			//�̹��� �߶󳻱�
			CvRect box;
			card_image_set = new IplImage*[card_num];
			for (int i = 0; i < card_num; i++) {

			//�߶� �̹���
			box.x = card_set[i].p2.x;
			box.y = card_set[i].p2.y;
			box.width = card_set[i].p1.x - card_set[i].p2.x + 1;
			box.height = card_set[i].p1.y - card_set[i].p2.y + 1;

			//�߶� �̹����� ���� �������� ����
			cvSetImageROI(g_image, box);

			//���� ���� �߶󳻱�
			card_image_set[i] = cvCreateImage(cvGetSize(g_image), g_image->depth, g_image->nChannels);
			cvCopy(g_image, card_image_set[i], NULL);

			//���� ���� �ʱ�ȭ
			cvResetImageROI(g_image);

			//�̹��� ���
			cvNamedWindow(name[i]);
			cvShowImage(name[i], card_image_set[i]);
			}

			//�޸� ����

			for (int i = 0; i < card_num; i++) {
			cvReleaseImage(&card_image_set[i]);
			}
			delete[] card_image_set;

			*/

			//�޸� ����
			delete[] card_set;

			for (int i = 0; i < (rows + 2); i++) {
				delete[] * (map + i);
				delete[] * (visit + i);
				delete[] * (temp_visit + i);
			}
			delete[] map;
			delete[] visit;
			delete[] temp_visit;
		}
		else {	//ī�� ������ 2�� �̸��̰ų� 20�� �ʰ��� ���

				//��� edge mark
			cvCopy(g_image, g_marked_original);
			for (int i = 0; i < card_num; i++) {
				cvDrawRect(g_marked_original, CvPoint(card_set[i].p1.x, card_set[i].p1.y), CvPoint(card_set[i].p2.x, card_set[i].p2.y), cvScalar(0, 0, 255), 2);
			}
			delete[] card_set;
		}

		//�̹��� ���
		cvShowImage("Original", g_image);
		cvShowImage("Binary", g_binary);
		cvShowImage("Contours", g_gray);
		cvShowImage("Marked Original", g_marked_original);

		//�޸� ����
		cvReleaseImage(&g_binary);
		cvReleaseImage(&g_gray);
		cvReleaseImage(&g_marked_original);

		cvReleaseMemStorage(&g_storage);
	}

	else return;
}

//label�� ���� �ٸ� color�� ��ȯ�ϴ� �Լ�
CvScalar color_select(int label) {
	switch (label) {
	case 1:
		return CvScalar(255, 0, 0);
	case 2:
		return CvScalar(0, 255, 0);
	case 3:
		return CvScalar(0, 0, 255);
	case 4:
		return CvScalar(255, 255, 0);
	case 5:
		return CvScalar(255, 0, 255);
	case 6:
		return CvScalar(0, 255, 255);
	case 7:
		return CvScalar(128, 64, 128);
	case 8:
		return CvScalar(128, 0, 0);
	case 9:
		return CvScalar(0, 128, 0);
	case 10:
		return CvScalar(0, 0, 128);
	default:
		return CvScalar(0, 0, 0);
	}
}

//ī�带 ��Ī�ϴ� �Լ�(label ����)
void cardMatch(Card card_set[], int card_num) {

	int label_num = 1, max_index;
	double max, result;

	//���ø� ��Ī�� �̿��ؼ� label ����
	for (int i = 0; i < card_num; i++) {
		if (card_set[i].label == 0) {
			max = 0;
			max_index = i;
			for (int j = i + 1; j < card_num; j++) {
				if (card_set[j].label == 0) {

					result = templMatch(card_set[i], card_set[j]);

					if (result >= max) {
						max = result;
						max_index = j;
					}
				}
			}

			card_set[i].label = label_num;
			card_set[max_index].label = label_num;
			label_num++;
		}
	}
}

//���ø� ��Ī �Լ�
double templMatch(Card &card1, Card &card2) {

	int window_width, window_height;
	int count;
	double rate = 0, max_rate = 0;
	CvScalar s1, s2;

	//RGB �� intensity�� ���� ����
	int thresh = 40;

	//*****  �� 4 ���� ���  *****

	//card1�� width, height ��� ũ�ų� ���� ��
	if (card1.width >= card2.width && card1.height >= card2.height) {
		//window�� width�� height�� ���� ������ ����
		window_width = card2.width;
		window_height = card2.height;

		for (int i = 0; i < card1.width - card2.width + 1; i++) {
			for (int j = 0; j < card1.height - card2.height + 1; j++) {
				count = 0;
				for (int x = 0; x < window_width; x++) {
					for (int y = 0; y < window_height; y++) {
						if (j + y + card1.p2.y < g_image->height && i + x + card1.p2.x < g_image->width) {

							s1 = cvGet2D(g_image, j + y + card1.p2.y, i + x + card1.p2.x);
							s2 = cvGet2D(g_image, y + card2.p2.y, x + card2.p2.x);

							if ((s1.val[0] - s2.val[0]) < thresh && (s1.val[1] - s2.val[1]) < thresh && (s1.val[2] == s2.val[2]) < thresh)
								count++;
						}
					}
				}
				rate = (double)count / (window_width * window_height);
				if (rate > max_rate)
					max_rate = rate;
			}
		}
		//���� ū ���絵 ��ȯ
		return max_rate;
	}
	//card2�� width, height ��� ũ�ų� ���� ��
	else if (card1.width <= card2.width && card1.height <= card2.height) {
		//window�� width�� height�� ���� ������ ����
		window_width = card1.width;
		window_height = card1.height;

		for (int i = 0; i < card2.width - card1.width + 1; i++) {
			for (int j = 0; j < card2.height - card1.height + 1; j++) {
				count = 0;
				for (int x = 0; x < window_width; x++) {
					for (int y = 0; y < window_height; y++) {
						if (j + y + card2.p2.y < g_image->height && i + x + card2.p2.x < g_image->width) {
							s1 = cvGet2D(g_image, y + card1.p2.y, x + card1.p2.x);
							s2 = cvGet2D(g_image, j + y + card2.p2.y, i + x + card2.p2.x);

							if ((s1.val[0] - s2.val[0]) < thresh && (s1.val[1] - s2.val[1]) < thresh && (s1.val[2] == s2.val[2]) < thresh)
								count++;
						}
					}
				}
				rate = (double)count / (window_width * window_height);
				if (rate > max_rate)
					max_rate = rate;
			}
		}
		//���� ū ���絵 ��ȯ
		return max_rate;
	}
	//card1�� width�� ũ�ų� ������, height�� ���� ��
	else if (card1.width >= card2.width && card1.height < card2.height) {
		//window�� width�� height�� ���� ������ ����
		window_width = card2.width;
		window_height = card1.height;

		for (int i = 0; i < card1.width - card2.width + 1; i++) {
			for (int j = 0; j < card2.height - card1.height + 1; j++) {
				count = 0;
				for (int x = 0; x < window_width; x++) {
					for (int y = 0; y < window_height; y++) {
						if (j + y + card2.p2.y < g_image->height && i + x + card1.p2.x < g_image->width) {
							s1 = cvGet2D(g_image, y + card1.p2.y, i + x + card1.p2.x);
							s2 = cvGet2D(g_image, j + y + card2.p2.y, x + card2.p2.x);

							if ((s1.val[0] - s2.val[0]) < thresh && (s1.val[1] - s2.val[1]) < thresh && (s1.val[2] == s2.val[2]) < thresh)
								count++;
						}
					}
				}
				rate = (double)count / (window_width * window_height);
				if (rate > max_rate)
					max_rate = rate;
			}
		}
		//���� ū ���絵 ��ȯ
		return max_rate;

	}
	//card1�� height�� ũ�ų� ������, width�� ���� ��
	else {
		//window�� width�� height�� ���� ������ ����
		window_width = card1.width;
		window_height = card2.height;

		for (int i = 0; i < card2.width - card1.width + 1; i++) {
			for (int j = 0; j < card2.height - card1.height + 1; j++) {
				count = 0;
				for (int x = 0; x < window_width; x++) {
					for (int y = 0; y < window_height; y++) {
						if (j + y + card1.p2.y < g_image->height && i + x + card2.p2.x < g_image->width) {
							s1 = cvGet2D(g_image, j + y + card1.p2.y, x + card1.p2.x);
							s2 = cvGet2D(g_image, y + card2.p2.y, i + x + card2.p2.x);

							if ((s1.val[0] - s2.val[0]) < thresh && (s1.val[1] - s2.val[1]) < thresh && (s1.val[2] == s2.val[2]) < thresh)
								count++;
						}
					}
				}
				rate = (double)count / (window_width * window_height);
				if (rate > max_rate)
					max_rate = rate;
			}
		}
		//���� ū ���絵 ��ȯ
		return max_rate;
	}
}

//visit �迭 �ʱ�ȭ�ϴ� �Լ�
void Initialize_visit_array(int height, int width) {
	for (int i = 0; i < height; i++)
		for (int k = 0; k < width; k++)
			visit[i][k] = 0;
}

//temp_visit �迭 �ʱ�ȭ�ϴ� �Լ�
void Initialize_temp_visit_array(int height, int width) {
	for (int i = 0; i < height; i++)
		for (int k = 0; k < width; k++)
			temp_visit[i][k] = 0;
}

//(x,y)Ÿ�Ͽ��� �����¿�� ���������� character�� ���� Ÿ���� �ִ��� �˻�.
bool To_up(int x, int y, int cardnum) {
	x--;

	for (; x >= 0; x--) {
		if (map[x][y] == -1)
			continue;
		if (map[x][y] != cardnum || temp_visit[x][y] == 1 || visit[x][y] == 1)
			return false;
		if (map[x][y] == cardnum && temp_visit[x][y] == 0 && visit[x][y] == 0) {
			temp_visit[x][y] = 1;
			return true;
		}
	}
	return false; //for �ȿ��� ������ �ȉ�ٸ� �� ã�� ��.
}

bool To_down(int x, int y, int cardnum, int height) {
	x++;

	for (; x <= (height - 1); x++) {
		if (map[x][y] == -1)
			continue;
		if (map[x][y] != cardnum || temp_visit[x][y] == 1 || visit[x][y] == 1)
			return false;
		if (map[x][y] == cardnum && temp_visit[x][y] == 0 && visit[x][y] == 0) {
			temp_visit[x][y] = 1;
			return true;

		}
	}
	return false; //for �ȿ��� ������ �ȉ�ٸ� �� ã�� ��.
}
bool To_right(int x, int y, int cardnum, int width) {
	y++;

	for (; y <= (width - 1); y++) {
		if (map[x][y] == -1)
			continue;
		if (map[x][y] != cardnum || temp_visit[x][y] == 1 || visit[x][y] == 1)
			return false;
		if (map[x][y] == cardnum && temp_visit[x][y] == 0 && visit[x][y] == 0) {
			temp_visit[x][y] = 1;
			return true;
		}
	}
	return false; //for �ȿ��� ������ �ȉ�ٸ� �� ã�� ��.
}
bool To_left(int x, int y, int cardnum) {
	y--;

	for (; y >= 0; y--) {
		if (map[x][y] == -1)
			continue;
		if (map[x][y] != cardnum || temp_visit[x][y] == 1 || visit[x][y] == 1)
			return false;
		if (map[x][y] == cardnum && temp_visit[x][y] == 0 && visit[x][y] == 0) {
			temp_visit[x][y] = 1;
			return true;
		}
	}
	return false; //for �ȿ��� ������ �ȉ�ٸ� �� ã�� ��.
}

//���� ������ pair�� ã�� ã�Ҵٸ� �� ī�� pair��ȣ�� ��ȯ ���� ���, 2��ī��� 20��ī�尡 ����ī���̰� �� ī����� ��� ��ȣ�� 1�̶� ����. 
//2��ī��� 20��ī�尡 ���� �����ϴٸ� ����ȣ 1 ��ȯ.
int find_pair(int height, int width) {

	Initialize_visit_array(height, width);
	Initialize_temp_visit_array(height, width);

	for (int i = 1; i <= (height - 2); i++) {   //(-2 �� ���� : ù ��,���� ������ ��,���� ��¥�� ��(.)�� �ԷµǹǷ�.)
		for (int k = 1; k <= (width - 2); k++) {
			if (map[i][k] == -1)   //-1�̶��, �� ���̶��
				continue;
			else if (map[i][k] != -1) {   //��(-1)�� �ƴ� � ī�� �󺧹�ȣ�� �߰��ߴٸ�,
				visit[i][k] = 1;
				temp_visit[i][k] = 1;

				////////////// ���� ��Ʈ //////////////
				if (To_up(i, k, map[i][k])) return map[i][k];
				if (To_right(i, k, map[i][k], width)) return map[i][k];
				if (To_down(i, k, map[i][k], height)) return map[i][k];
				if (To_left(i, k, map[i][k])) return map[i][k];
				////////////// ���� ��Ʈ �� //////////////

				////////////// �� �� ���� ��Ʈ //////////////
				for (int t = i - 1; t >= 0; t--) {   //up - RIGHT & LEFT
					if (map[t][k] != -1)
						break;
					if (To_right(t, k, map[i][k], width)) return map[i][k];
					if (To_left(t, k, map[i][k])) return map[i][k];

				}
				for (int t = i + 1; t <= (height - 1); t++) {   //down - RIGHT & LEFT
					if (map[t][k] != -1)
						break;
					if (To_right(t, k, map[i][k], width)) return map[i][k];
					if (To_left(t, k, map[i][k])) return map[i][k];
				}
				for (int r = k + 1; r <= (width - 1); r++) {   //right - UP & DOWN
					if (map[i][r] != -1)
						break;
					if (To_up(i, r, map[i][k])) return map[i][k];
					if (To_down(i, r, map[i][k], height)) return map[i][k];
				}
				for (int r = k - 1; r >= 0; r--) {   //left - UP & DOWN
					if (map[i][r] != -1)
						break;
					if (To_up(i, r, map[i][k])) return map[i][k];
					if (To_down(i, r, map[i][k], height)) return map[i][k];
				}
				////////////// �� �� ���� ��Ʈ �� //////////////

				////////////// �� �� ���� ��Ʈ //////////////
				for (int t = i - 1; t >= 0; t--) {   //up - right - UP & DOWN
					if (map[t][k] != -1)
						break;
					for (int r = k + 1; r != (width - 1); r++) {
						if (map[t][r] != -1)
							break;
						if (To_up(t, r, map[i][k])) return map[i][k];
						if (To_down(t, r, map[i][k], height)) return map[i][k];
					}
				}
				for (int t = i - 1; t >= 0; t--) {   //up - left - UP & DOWN
					if (map[t][k] != -1)
						break;
					for (int r = k - 1; r != 0; r--) {
						if (map[t][r] != -1)
							break;
						if (To_up(t, r, map[i][k])) return map[i][k];
						if (To_down(t, r, map[i][k], height)) return map[i][k];
					}
				}
				for (int r = k + 1; r <= (width - 1); r++) {   //right - up - RIGHT & LEFT
					if (map[i][r] != -1)
						break;
					for (int t = i - 1; t != 0; t--) {
						if (map[t][r] != -1)
							break;
						if (To_right(t, r, map[i][k], width)) return map[i][k];
						if (To_left(t, r, map[i][k])) return map[i][k];
					}
				}
				for (int r = k + 1; r <= (width - 1); r++) {   //right - down - RIGHT & LEFT
					if (map[i][r] != -1)
						break;
					for (int t = i + 1; t != (height - 1); t++) {
						if (map[t][r] != -1)
							break;
						if (To_right(t, r, map[i][k], width)) return map[i][k];
						if (To_left(t, r, map[i][k])) return map[i][k];
					}
				}
				for (int t = i + 1; t <= (height - 1); t++) {   //down - right - UP & DOWN
					if (map[t][k] != -1)
						break;
					for (int r = k + 1; r != (width - 1); r++) {
						if (map[t][r] != -1)
							break;
						if (To_up(t, r, map[i][k])) return map[i][k];
						if (To_down(t, r, map[i][k], height)) return map[i][k];
					}
				}
				for (int t = i + 1; t <= (height - 1); t++) {   //down - left - UP & DOWN
					if (map[t][k] != -1)
						break;
					for (int r = k - 1; r != 0; r--) {
						if (map[t][r] != -1)
							break;
						if (To_up(t, r, map[i][k])) return map[i][k];
						if (To_down(t, r, map[i][k], height)) return map[i][k];
					}
				}
				for (int r = k - 1; r >= 0; r--) {   //left - up - RIGHT & LEFT
					if (map[i][r] != -1)
						break;
					for (int t = i - 1; t != 0; t--) {
						if (map[t][r] != -1)
							break;
						if (To_right(t, r, map[i][k], width)) return map[i][k];
						if (To_left(t, r, map[i][k])) return map[i][k];
					}
				}
				for (int r = k - 1; r >= 0; r--) {   //left - down - RIGHT & LEFT
					if (map[i][r] != -1)
						break;
					for (int t = i + 1; t != (height - 1); t++) {
						if (map[t][r] != -1)
							break;
						if (To_right(t, r, map[i][k], width)) return map[i][k];
						if (To_left(t, r, map[i][k])) return map[i][k];
					}
				}
				////////////// �� �� ���� ��Ʈ �� //////////////

				Initialize_temp_visit_array(height, width);   //temp_visit �迭 �ʱ�ȭ
			}
		}
	}

	return -2;      //�� ã�� ���
}

//��Ī�Ǵ� ī�带 �����ϴ� �Լ�
void highlight(Card* card_set, int index) {

	CvScalar s;

	//����� minus_val ��ŭ ��⸦ ���ҽ�Ű��, ã�� ī��� plus_val��ŭ ��� ����
	int minus_val = 100, plus_val = 30;

	for (int x = 0; x < g_marked_original->width; x++) {
		for (int y = 0; y < g_marked_original->height; y++) {
			if (x < card_set[index].p2.x || x > card_set[index].p1.x || y < card_set[index].p2.y || y > card_set[index].p1.y) {
				s = cvGet2D(g_marked_original, y, x);
				s.val[0] -= minus_val;
				if (s.val[0] < 0)
					s.val[0] = 0;
				s.val[1] -= minus_val;
				if (s.val[1] < 0)
					s.val[1] = 0;
				s.val[2] -= minus_val;
				if (s.val[2] < 0)
					s.val[2] = 0;
				cvSet2D(g_marked_original, y, x, s);
			}
			else {

				s = cvGet2D(g_marked_original, y, x);
				s.val[0] += plus_val;
				if (s.val[0] > 255)
					s.val[0] = 255;
				s.val[1] += plus_val;
				if (s.val[1] > 255)
					s.val[1] = 255;
				s.val[2] += plus_val;
				if (s.val[2] > 255)
					s.val[2] = 255;
				cvSet2D(g_marked_original, y, x, s);
			}
		}
	}
	cvDrawRect(g_marked_original, CvPoint(card_set[index].p1.x, card_set[index].p1.y), CvPoint(card_set[index].p2.x, card_set[index].p2.y), CvScalar(0, 0, 255), 3);

}

//main �Լ�
int main(int argc, char **argv)
{
	//ȭ�� ĸ��
	HWND hwndDesktop = GetDesktopWindow();
	int key = 0;

	cvNamedWindow("Original", 1);
	cvNamedWindow("Binary", 1);
	cvNamedWindow("Contours", 1);
	cvNamedWindow("Marked Original", 1);

	//������ʹ� �ǽð����� Ʈ��ŷ�ϴ� ���

	//�� ���� �Է� ���� ��, ����
	char c;
	scanf("%c", &c);

	//�ð� ��� ���� ����
	clock_t begin, end;

	Mat src;

	//key�� 27, �� esc�� ���� ������ �ݺ�
	while (key != 27)
	{
		begin = clock();

		//ȭ�� ĸ�ĸ� Mat ���Ŀ� ����
		src = hwnd2mat(hwndDesktop);

		//Mat ������ IplImage* �������� ��ȯ
		g_image = &IplImage(src);

		//���� g_image�� NULL�̶��
		if (!g_image) {
			//���� �޽��� ���
			printf("Faield to open image\n");
			return -1;
		}

		//���� ���� ����
		cvSetImageROI(g_image, cvRect(200, 250, 700, 500));

		//Ʈ���� ����
		cvCreateTrackbar("Threshold", "Contours", &g_thresh, 255, on_trackbar);

		//�Լ� ȣ��
		on_trackbar(0);

		//���� ���� �ʱ�ȭ
		cvResetImageROI(g_image);

		end = clock();

		//�� �����Ӵ� �ɸ� �ҿ� �ð� ���
		cout << "Elapsed Time : " << (end - begin) << endl;

		//Ű �Է� ��� �ð� (100 ms)
		key = waitKey(100);

		//Mat ���� �޸� ����
		src.release();
	}

	//������ʹ� �̹��� �ϳ��� ó���ϴ� ���

	/*
	//�̹��� �ҷ�����
	g_image = cvLoadImage("test3.jpg");

	//���� g_image�� NULL�̶��
	if (!g_image) {
	//���� �޽��� ���
	printf("Faield to open image\n");
	return -1;
	}

	//�̹��� ���
	cvShowImage("Original", g_image);

	//Ʈ���� ����
	cvCreateTrackbar("Threshold", "Contours", &g_thresh, 255, on_trackbar);

	//�Լ� ȣ��
	on_trackbar(0);

	//Ű �Է� ������ ����
	waitKey(0);
	*/

	//��� ������ ����
	cvDestroyAllWindows();
}
