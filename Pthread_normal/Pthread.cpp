#include<iostream>
#include<Windows.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pthread.h>
#include<immintrin.h>
#include<semaphore.h>//�ź���������ͷ�ļ�
using namespace std;
alignas(16) float gdata[10000][10000];//���ж������
float gdata2[10000][10000];
float gdata1[10000][10000];
float gdata3[10000][10000];
int n;
//�����߳����ݽṹ
typedef struct {
	int t_id;	//�̱߳��
}threadParam_t;
const int Num_thread = 8;//8��CPU

sem_t sem_leader;		//���г���������̵߳��ź���
sem_t sem_Division[Num_thread - 1];//�жϳ����Ƿ���ɵ��ź���
sem_t sem_Elimination[Num_thread - 1];//�Ƿ���Խ�����һ����ȥ���ź���


//�̺߳���1
void *threadFunc_SSE(void* Param)//����void���͵�ָ���ָ���������͵�����
{
	threadParam_t* p = (threadParam_t*)Param;//ǿ������ת�����߳����ݽṹ
	int t_id = p->t_id;//�̵߳ı�Ż�ȡ
	//��һ���̸߳���������������������Ҳ����к�����Ԫ����
	for (int k = 0; k < n; k++)
	{
		__m128 r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
		if (t_id == 0)
		{
			float temp[4] = { gdata3[k][k],gdata3[k][k],gdata3[k][k],gdata3[k][k] };
			r0 = _mm_loadu_ps(temp);//�ڴ治�������ʽ���ص������Ĵ�����
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = _mm_loadu_ps(gdata3[k] + j);
				r1 = _mm_div_ps(r1, r0);//������������λ���
				_mm_storeu_ps(gdata3[k], r1);//���������·Ż��ڴ�
			}
			//��ʣ�಻��4�������ݽ�����Ԫ
			for (j; j < n; j++)
			{
				gdata3[k][j] = gdata3[k][j] / gdata3[k][k];
			}
			gdata3[k][k] = 1.0;
		}
		else {
			sem_wait(&sem_Division[t_id - 1]);//����ֱ�������������
		}
		if (t_id == 0)//����������������������߳�
		{
			for (int i = 0; i < Num_thread-1; i++)
			{
				sem_post(&sem_Division[i]);
			}
		}
		//���񻮷֣�����Ϊ��λ���л���
		
		for (int i = k + 1 + t_id; i < n; i += Num_thread)
		{
			//�����������ѭ�����SIMD����
			float temp2[4] = { gdata3[i][k],gdata3[i][k],gdata3[i][k],gdata3[i][k] };
			r0 = _mm_loadu_ps(temp2);
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = _mm_loadu_ps(gdata3[k] + j);
				r2 = _mm_loadu_ps(gdata3[i] + j);
				r3 = _mm_mul_ps(r0, r1);
				r2 = _mm_sub_ps(r2, r3);
				_mm_storeu_ps(gdata3[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata3[i][j] = gdata3[i][j] - (gdata3[i][k] * gdata3[k][j]);
			}
			gdata3[i][k] = 0;
		}
		if (t_id == 0)
		{
			for (int i = 0; i < Num_thread-1; i++)
				sem_wait(&sem_leader);//���̵߳ȴ������߳������ȥ
			for (int i = 0; i < Num_thread - 1; i++)
				sem_post(&sem_Elimination[i]);//�������������ȥ��֪ͨ������һ������
		}
		else {
			sem_post(&sem_leader);//�����߳�֪ͨleader�������ȥ
			sem_wait(&sem_Elimination[t_id - 1]);//�ȴ�������һ�ֵ�֪ͨ
		}
	}
	pthread_exit(NULL);
	return NULL;
}

//�̺߳���2
void* threadFunc_AVX(void* Param)//����void���͵�ָ���ָ���������͵�����
{
	threadParam_t* p = (threadParam_t*)Param;//ǿ������ת�����߳����ݽṹ
	int t_id = p->t_id;//�̵߳ı�Ż�ȡ
	//��һ���̸߳���������������������Ҳ����к�����Ԫ����
	for (int k = 0; k < n; k++)
	{
		__m256 r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
		if (t_id == 0)
		{
			float temp[8] = { gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k],gdata2[k][k] };
			r0 = _mm256_load_ps(temp);//�ڴ�������ʽ���ص������Ĵ�����
			int j;
			for (j = k + 1; j + 8 <= n; j += 8)
			{
				r1 = _mm256_load_ps(gdata2[k] + j);
				r1 = _mm256_div_ps(r1, r0);//������������λ���
				_mm256_store_ps(gdata2[k], r1);//���������·Ż��ڴ�
			}
			//��ʣ�಻��8�������ݽ�����Ԫ
			for (j; j < n; j++)
			{
				gdata2[k][j] = gdata2[k][j] / gdata2[k][k];
			}
			gdata2[k][k] = 1.0;
		}
		else {
			sem_wait(&sem_Division[t_id - 1]);//����ֱ�������������
		}
		if (t_id == 0)//����������������������߳�
		{
			for (int i = 0; i < Num_thread - 1; i++)
			{
				sem_post(&sem_Division[i]);
			}
		}
		//���񻮷֣�����Ϊ��λ���л���

		for (int i = k + 1 + t_id; i < n; i += Num_thread)
		{
			//�����������ѭ�����SIMD����
			float temp2[8] = { gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k],gdata2[i][k] };
			r0 = _mm256_load_ps(temp2);
			int j;
			for (j = k + 1; j + 8 <= n; j += 8)
			{
				r1 = _mm256_load_ps(gdata2[k] + j);
				r2 = _mm256_load_ps(gdata2[i] + j);
				r3 = _mm256_mul_ps(r0, r1);
				r2 = _mm256_sub_ps(r2, r3);
				_mm256_store_ps(gdata2[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata2[i][j] = gdata2[i][j] - (gdata2[i][k] * gdata2[k][j]);
			}
			gdata2[i][k] = 0;
		}
		if (t_id == 0)
		{
			for (int i = 0; i < Num_thread - 1; i++)
				sem_wait(&sem_leader);//���̵߳ȴ������߳������ȥ
			for (int i = 0; i < Num_thread - 1; i++)
				sem_post(&sem_Elimination[i]);//�������������ȥ��֪ͨ������һ������
		}
		else {
			sem_post(&sem_leader);//�����߳�֪ͨleader�������ȥ
			sem_wait(&sem_Elimination[t_id - 1]);//�ȴ�������һ�ֵ�֪ͨ
		}
	}
	pthread_exit(NULL);
	return NULL;
}

//���ݳ�ʼ��
void Initialize(int N)
{
	for (int i = 0; i < N; i++)
	{
		//���Ƚ�ȫ��Ԫ����Ϊ0���Խ���Ԫ����Ϊ1
		for (int j = 0; j < N; j++)
		{
			gdata[i][j] = 0;
			gdata1[i][j] = 0;
			gdata2[i][j] = 0;
			gdata3[i][j] = 0;
		}
		gdata[i][i] = 1.0;
		//�������ǵ�λ�ó�ʼ��Ϊ�����
		for (int j = i + 1; j < N; j++)
		{
			gdata[i][j] = rand();
			gdata1[i][j] = gdata[i][j] = gdata2[i][j] = gdata3[i][j];
		}
	}
	for (int k = 0; k < N; k++)
	{
		for (int i = k + 1; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				gdata[i][j] += gdata[k][j];
				gdata1[i][j] += gdata1[k][j];
				gdata2[i][j] += gdata2[k][j];
				gdata3[i][j] += gdata3[k][j];
			}
		}
	}

}
//ƽ���㷨
void Normal_alg(int N)
{
	int i, j, k;
	for (k = 0; k < N; k++)
	{
		for (j = k + 1; j < N; j++)
		{
			gdata1[k][j] = gdata1[k][j] / gdata1[k][k];
		}
		gdata1[k][k] = 1.0;
		for (i = k + 1; i < N; i++)
		{
			for (j = k + 1; j < N; j++)
			{
				gdata1[i][j] = gdata1[i][j] - (gdata1[i][k] * gdata1[k][j]);
			}
			gdata1[i][k] = 0;
		}
	}
}

//��ȫ������SIMD�Ż�
void Par_alg_all(int n)
{
	int i, j, k;
	__m128 r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
	for (k = 0; k < n; k++)
	{
		float temp[4] = { gdata[k][k],gdata[k][k],gdata[k][k],gdata[k][k] };
		r0 = _mm_loadu_ps(temp);//�ڴ治�������ʽ���ص������Ĵ�����
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			r1 = _mm_loadu_ps(gdata[k] + j);
			r1 = _mm_div_ps(r1, r0);//������������λ���
			_mm_storeu_ps(gdata[k], r1);//���������·Ż��ڴ�
		}
		//��ʣ�಻��4�������ݽ�����Ԫ
		for (j; j < n; j++)
		{
			gdata[k][j] = gdata[k][j] / gdata[k][k];
		}
		gdata[k][k] = 1.0;
		//���϶�Ӧ������һ������ѭ���Ż���SIMD

		for (i = k + 1; i < n; i++)
		{
			float temp2[4] = { gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k] };
			r0 = _mm_loadu_ps(temp2);
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = _mm_loadu_ps(gdata[k] + j);
				r2 = _mm_loadu_ps(gdata[i] + j);
				r3 = _mm_mul_ps(r0, r1);
				r2 = _mm_sub_ps(r2, r3);
				_mm_storeu_ps(gdata[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata[i][j] = gdata[i][j] - (gdata[i][k] * gdata[k][j]);
			}
			gdata[i][k] = 0;
		}
	}
}

void pthread_SSE()
{
	//��ʼ���ź���
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < Num_thread - 1; i++)
	{
		sem_init(&sem_Division[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	pthread_t* handles = new pthread_t[Num_thread];//�����߳̾��
	threadParam_t* param = new threadParam_t[Num_thread];//��Ҫ���ݵĲ������
	for (int t_id = 0; t_id < Num_thread; t_id++)
	{
		param[t_id].t_id = t_id;//���̲߳������ݣ��߳�����
		pthread_create(&handles[t_id], NULL, threadFunc_SSE, &param[t_id]);//�����̺߳���
	}
	for (int t_id = 0; t_id < Num_thread; t_id++)
		pthread_join(handles[t_id], NULL);
	sem_destroy(sem_Division);
	sem_destroy(sem_Elimination);
	sem_destroy(&sem_leader);
}

void pthread_AVX()
{
	//��ʼ���ź���
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < Num_thread - 1; i++)
	{
		sem_init(&sem_Division[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	pthread_t* handles = new pthread_t[Num_thread];//�����߳̾��
	threadParam_t* param = new threadParam_t[Num_thread];//��Ҫ���ݵĲ������
	for (int t_id = 0; t_id < Num_thread; t_id++)
	{
		param[t_id].t_id = t_id;//���̲߳������ݣ��߳�����
		pthread_create(&handles[t_id], NULL, threadFunc_AVX, &param[t_id]);//�����̺߳���
	}
	for (int t_id = 0; t_id < Num_thread; t_id++)
		pthread_join(handles[t_id], NULL);
	sem_destroy(sem_Division);
	sem_destroy(sem_Elimination);
	sem_destroy(&sem_leader);
}


int main()
{
	LARGE_INTEGER fre, begin, end;
	double gettime;
	cin >> n;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Initialize(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "intial time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Normal_alg(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "normal time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	pthread_SSE();
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "pthread_SSE time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	pthread_AVX();
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "pthread_AVX time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Par_alg_all(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "SIMD time: " << gettime << " ms" << endl;
}
