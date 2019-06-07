
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <Windows.h>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <time.h>

using namespace std;
using namespace cv;

// 카드 구조체
typedef struct Card {
	CvPoint p1;      // (max_x, max_y)
	CvPoint p2;      // (min_x, min_y)
	int width;
	int height;
	int label;
}Card;

IplImage* g_image = NULL;			//원본 이미지
IplImage* g_gray = NULL;			//gray scale된 이미지
IplImage* g_binary = NULL;			//edge 이미지
IplImage* g_marked_original = NULL;	//Marking된 이미지
IplImage** card_image_set = NULL;	//카드 이미지 집합
CvMemStorage* g_storage = NULL;		//저장소

int g_thresh = 180;      //Edge detecting 경계
int card_num;				//카드 개수
int** map;					//맵
int **visit, **temp_visit;	//사천성 알고리즘 변수
int prev_card_num = 0;		//이전 프레임에서의 카드 개수

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

//HWND 형식을 Mat으로 바꿔주는 함수
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

	RECT windowsize;    // 스크린의 height와 weight 가져오는 변수
	GetClientRect(hwnd, &windowsize);

	srcheight = windowsize.bottom;
	srcwidth = windowsize.right;
	height = windowsize.bottom;
	width = windowsize.right;

	//스크린 전체 추출
	src.create(height, width, CV_8UC4);

	//비트맵 생성
	hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
	bi.biSize = sizeof(BITMAPINFOHEADER);
	bi.biWidth = width;
	bi.biHeight = -height;  //그리는 방향은 거꾸로
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 0;
	bi.biXPelsPerMeter = 0;
	bi.biYPelsPerMeter = 0;
	bi.biClrUsed = 0;
	bi.biClrImportant = 0;

	//위에서 만든 장치와 비트맵 연결
	SelectObject(hwindowCompatibleDC, hbwindow);
	//윈도우 장치의 정보를 비트맵에 복사
	StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, SRCCOPY);
	GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO *)&bi, DIB_RGB_COLORS);

	//메모리 해제
	DeleteObject(hbwindow);
	DeleteDC(hwindowCompatibleDC);
	ReleaseDC(hwnd, hwindowDC);

	return src;
}

//트랙바에 따라서 영상 처리하는 함수
void on_trackbar(int pos) {

	//초기화 작업
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

	//외곽선 찾기
	CvSeq* contours = 0;

	card_num = cvFindContours(   //num에는 외곽선 개수가 저장
		g_gray,               //입력영상
		g_storage,            //검출된 외곽선을 기록하기 위한 메모리 스토리지
		&contours,            //외곽선의 좌표들이 저장된 Sequence
		sizeof(CvContour),
		CV_RETR_EXTERNAL      //가장 바깥의 외곽선만 표시
	);

	//이미지를 지움 (외곽선만 그리기 위해서)
	cvZero(g_gray);

	if (contours) {
		//외곽선을 찾은 정보(contour)를 이용하여 외곽선을 그림
		cvDrawContours(
			g_gray,                //외곽선이 그려질 영상
			contours,              //외곽선 트리의 루트노드
			cvScalarAll(255),      //외부 외곽선의 색상
			cvScalarAll(128),      //내부 외곽선의 색상
			100                    //외곽선을 그릴때 이동할 깊이
		);
	}

	//불필요한 연산을 줄이기 위해서 카드 개수가 2개 이상이고, 이전 프레임의 카드 개수와 다를 경우만 연산 (아닌 경우 그냥 return)
	if (prev_card_num != card_num && card_num >= 2) {

		//외곽선의 개수만큼 카드 생성
		Card* card_set = new Card[card_num];

		//외곽선의 정보를 이용해서 카드 위치 계산
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

		//메모리 해제
		cvClearSeq(contours);

		int max_size = 0, max_index = 0;

		//width, height, max_size, max_index 계산
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

		//불필요한 연산을 줄이기 위해서 의미 없는 edge들 체크
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

		//체크된 edge들 무시
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

		//메모리 해제
		delete[] check;
		delete[] card_set;

		//의미 있는 edge들로 카드 set 재구성
		card_set = temp;

		//이전 카드 개수 갱신
		prev_card_num = card_num;

		//카드 개수 20개로 한정
		//제외한 edge들을 제외하고 2개 이상 20개 이하일 때만 수행 (아닐 경우 매칭하지 않고 모든 edge들 mark)
		if (card_num <= 20 && card_num >= 2) {

			//카드 매칭
			cardMatch(card_set, card_num);

			int c_w, c_h; // card_width, card_height
			c_w = card_set[0].p1.x - card_set[0].p2.x;
			c_h = card_set[0].p1.y - card_set[0].p2.y;

			if (c_w == 0 || c_h == 0)
				return;

			//map_minx, map_miny, map_maxx, map_maxy 계산
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

			//카드크기 이용하여 맵 크기 구하고 배열로 변환
			int rows, cols;
			cols = (m_maxx - m_minx) / c_w;
			rows = (m_maxy - m_miny) / c_h;

			//길 확보를 위해서 1칸씩 양 옆으로 추가 할당

			map = (int**)malloc(sizeof(int*)*(rows + 2));
			for (int i = 0; i < (rows + 2); i++) {
				map[i] = (int*)malloc(sizeof(int)*(cols + 2));
			}
			for (int i = 0; i < rows + 2; i++)
				for (int j = 0; j < cols + 2; j++)
					map[i][j] = -1;

			//사천성 알고리즘을 위한 visit, temp_visit 할당
			visit = (int**)malloc(sizeof(int*)*(rows + 2));
			for (int i = 0; i < (rows + 2); i++) {
				visit[i] = (int*)malloc(sizeof(int)*(cols + 2));
			}
			temp_visit = (int**)malloc(sizeof(int*)*(rows + 2));
			for (int i = 0; i < (rows + 2); i++) {
				temp_visit[i] = (int*)malloc(sizeof(int)*(cols + 2));
			}

			//카드정보 배열에 넣기
			int div_w = (m_maxx - m_minx) / cols;
			int div_h = (m_maxy - m_miny) / rows;
			for (int i = 0; i < card_num; i++) {
				int x = 0, y = 0;
				int c_ix, c_iy; // 카드 x좌표 중점, 카드 y좌표 중점
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
				//원본 이미지에서 카드 위치에 마킹(하이라이팅)
				for (int i = 0; i < card_num; i++) {
					if (card_set[i].label == index) {
						highlight(card_set, i);
					}
				}
			}
			else {
				printf("Failed to find pair\n");
			}

			/*  각각의 카드들 이미지 따로 떼어서 출력함으로써 카드를 잘 인식하는지 확인하는 부분 (실제로 실행할 때는 없어도 되는 부분)

			//카드 이미지 20개 이름
			char* name[20] = { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t" };

			//이미지 잘라내기
			CvRect box;
			card_image_set = new IplImage*[card_num];
			for (int i = 0; i < card_num; i++) {

			//잘라낼 이미지
			box.x = card_set[i].p2.x;
			box.y = card_set[i].p2.y;
			box.width = card_set[i].p1.x - card_set[i].p2.x + 1;
			box.height = card_set[i].p1.y - card_set[i].p2.y + 1;

			//잘라낼 이미지를 관심 영역으로 지정
			cvSetImageROI(g_image, box);

			//관심 영역 잘라내기
			card_image_set[i] = cvCreateImage(cvGetSize(g_image), g_image->depth, g_image->nChannels);
			cvCopy(g_image, card_image_set[i], NULL);

			//관심 영역 초기화
			cvResetImageROI(g_image);

			//이미지 출력
			cvNamedWindow(name[i]);
			cvShowImage(name[i], card_image_set[i]);
			}

			//메모리 해제

			for (int i = 0; i < card_num; i++) {
			cvReleaseImage(&card_image_set[i]);
			}
			delete[] card_image_set;

			*/

			//메모리 해제
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
		else {	//카드 개수가 2개 미만이거나 20개 초과일 경우

				//모든 edge mark
			cvCopy(g_image, g_marked_original);
			for (int i = 0; i < card_num; i++) {
				cvDrawRect(g_marked_original, CvPoint(card_set[i].p1.x, card_set[i].p1.y), CvPoint(card_set[i].p2.x, card_set[i].p2.y), cvScalar(0, 0, 255), 2);
			}
			delete[] card_set;
		}

		//이미지 출력
		cvShowImage("Original", g_image);
		cvShowImage("Binary", g_binary);
		cvShowImage("Contours", g_gray);
		cvShowImage("Marked Original", g_marked_original);

		//메모리 해제
		cvReleaseImage(&g_binary);
		cvReleaseImage(&g_gray);
		cvReleaseImage(&g_marked_original);

		cvReleaseMemStorage(&g_storage);
	}

	else return;
}

//label에 따라 다른 color를 반환하는 함수
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

//카드를 매칭하는 함수(label 결정)
void cardMatch(Card card_set[], int card_num) {

	int label_num = 1, max_index;
	double max, result;

	//템플릿 매칭을 이용해서 label 결정
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

//템플릿 매칭 함수
double templMatch(Card &card1, Card &card2) {

	int window_width, window_height;
	int count;
	double rate = 0, max_rate = 0;
	CvScalar s1, s2;

	//RGB 각 intensity의 오차 범위
	int thresh = 40;

	//*****  총 4 가지 경우  *****

	//card1이 width, height 모두 크거나 같을 때
	if (card1.width >= card2.width && card1.height >= card2.height) {
		//window의 width와 height을 작은 쪽으로 선택
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
		//가장 큰 유사도 반환
		return max_rate;
	}
	//card2가 width, height 모두 크거나 같을 때
	else if (card1.width <= card2.width && card1.height <= card2.height) {
		//window의 width와 height을 작은 쪽으로 선택
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
		//가장 큰 유사도 반환
		return max_rate;
	}
	//card1이 width는 크거나 같은데, height는 작을 때
	else if (card1.width >= card2.width && card1.height < card2.height) {
		//window의 width와 height을 작은 쪽으로 선택
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
		//가장 큰 유사도 반환
		return max_rate;

	}
	//card1이 height는 크거나 같은데, width는 작을 때
	else {
		//window의 width와 height을 작은 쪽으로 선택
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
		//가장 큰 유사도 반환
		return max_rate;
	}
}

//visit 배열 초기화하는 함수
void Initialize_visit_array(int height, int width) {
	for (int i = 0; i < height; i++)
		for (int k = 0; k < width; k++)
			visit[i][k] = 0;
}

//temp_visit 배열 초기화하는 함수
void Initialize_temp_visit_array(int height, int width) {
	for (int i = 0; i < height; i++)
		for (int k = 0; k < width; k++)
			temp_visit[i][k] = 0;
}

//(x,y)타일에서 상하좌우로 움직여보며 character와 같은 타일이 있는지 검사.
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
	return false; //for 안에서 리턴이 안됬다면 못 찾은 것.
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
	return false; //for 안에서 리턴이 안됬다면 못 찾은 것.
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
	return false; //for 안에서 리턴이 안됬다면 못 찾은 것.
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
	return false; //for 안에서 리턴이 안됬다면 못 찾은 것.
}

//연결 가능한 pair를 찾고 찾았다면 그 카드 pair번호를 반환 예를 들어, 2번카드와 20번카드가 같은카드이고 이 카드들의 페어 번호는 1이라 하자. 
//2번카드와 20번카드가 연결 가능하다면 페어번호 1 반환.
int find_pair(int height, int width) {

	Initialize_visit_array(height, width);
	Initialize_temp_visit_array(height, width);

	for (int i = 1; i <= (height - 2); i++) {   //(-2 한 이유 : 첫 행,열과 마지막 행,열은 어짜피 점(.)만 입력되므로.)
		for (int k = 1; k <= (width - 2); k++) {
			if (map[i][k] == -1)   //-1이라면, 즉 길이라면
				continue;
			else if (map[i][k] != -1) {   //길(-1)이 아닌 어떤 카드 라벨번호를 발견했다면,
				visit[i][k] = 1;
				temp_visit[i][k] = 1;

				////////////// 직선 루트 //////////////
				if (To_up(i, k, map[i][k])) return map[i][k];
				if (To_right(i, k, map[i][k], width)) return map[i][k];
				if (To_down(i, k, map[i][k], height)) return map[i][k];
				if (To_left(i, k, map[i][k])) return map[i][k];
				////////////// 직선 루트 끝 //////////////

				////////////// 한 번 꺽는 루트 //////////////
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
				////////////// 한 번 꺽는 루트 끝 //////////////

				////////////// 두 번 꺽는 루트 //////////////
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
				////////////// 두 번 꺽는 루트 끝 //////////////

				Initialize_temp_visit_array(height, width);   //temp_visit 배열 초기화
			}
		}
	}

	return -2;      //못 찾은 경우
}

//매칭되는 카드를 강조하는 함수
void highlight(Card* card_set, int index) {

	CvScalar s;

	//배경은 minus_val 만큼 밝기를 감소시키고, 찾은 카드는 plus_val만큼 밝기 증가
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

//main 함수
int main(int argc, char **argv)
{
	//화면 캡쳐
	HWND hwndDesktop = GetDesktopWindow();
	int key = 0;

	cvNamedWindow("Original", 1);
	cvNamedWindow("Binary", 1);
	cvNamedWindow("Contours", 1);
	cvNamedWindow("Marked Original", 1);

	//여기부터는 실시간으로 트랙킹하는 기능

	//한 글자 입력 받은 후, 시작
	char c;
	scanf("%c", &c);

	//시간 재기 위한 변수
	clock_t begin, end;

	Mat src;

	//key가 27, 즉 esc가 눌릴 때까지 반복
	while (key != 27)
	{
		begin = clock();

		//화면 캡쳐를 Mat 형식에 저장
		src = hwnd2mat(hwndDesktop);

		//Mat 형식을 IplImage* 형식으로 변환
		g_image = &IplImage(src);

		//만약 g_image가 NULL이라면
		if (!g_image) {
			//에러 메시지 출력
			printf("Faield to open image\n");
			return -1;
		}

		//관심 영역 세팅
		cvSetImageROI(g_image, cvRect(200, 250, 700, 500));

		//트랙바 생성
		cvCreateTrackbar("Threshold", "Contours", &g_thresh, 255, on_trackbar);

		//함수 호출
		on_trackbar(0);

		//관심 영역 초기화
		cvResetImageROI(g_image);

		end = clock();

		//한 프레임당 걸린 소요 시간 출력
		cout << "Elapsed Time : " << (end - begin) << endl;

		//키 입력 대기 시간 (100 ms)
		key = waitKey(100);

		//Mat 변수 메모리 해제
		src.release();
	}

	//여기부터는 이미지 하나만 처리하는 기능

	/*
	//이미지 불러오기
	g_image = cvLoadImage("test3.jpg");

	//만약 g_image가 NULL이라면
	if (!g_image) {
	//에러 메시지 출력
	printf("Faield to open image\n");
	return -1;
	}

	//이미지 출력
	cvShowImage("Original", g_image);

	//트랙바 생성
	cvCreateTrackbar("Threshold", "Contours", &g_thresh, 255, on_trackbar);

	//함수 호출
	on_trackbar(0);

	//키 입력 받으면 종료
	waitKey(0);
	*/

	//모든 윈도우 제거
	cvDestroyAllWindows();
}
