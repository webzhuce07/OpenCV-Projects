#include <iostream>
#include <random>
#include <ctime>

using namespace std;

class AlgorithmPSO 
{
public:
	AlgorithmPSO()
	{

	}
	~AlgorithmPSO()
	{
		delete[] m_y;
		delete[] m_x;
		delete[] m_v;
		delete[] m_pbest;
	}

    //适应度计算函数，每个粒子都有它的适应度
     void fitnessFunction()
	 {
        for(int i=0;i<m_n;i++)
		{
            m_y[i] = -1.0f * m_x[i] * (m_x[i]-2.0);
        }
     }
    
	 //初始化
	 void init()
	 { 
        m_x = new double[m_n];
        m_v = new double[m_n];
        m_y = new double[m_n];
        m_pbest = new double[m_n];
        /***
         * 本来是应该随机产生的，为了方便演示，我这里手动随机落两个点，分别落在最大值两边
         */
        m_x[0] = 0.0;
        m_x[1] = 2.0;
        m_v[0] = 0.01;
        m_v[1] = 0.02;
        fitnessFunction();
        //初始化当前个体最优位置，并找到群体最优位置
        for(int i=0;i<m_n;i++)
		{
            m_pbest[i] = m_y[i];
            if(m_y[i]>gbest) gbest=m_y[i];
        }
       std::cout << ("Algorithm Starting , gbest:") << gbest << std::endl;
    }

    double getMAX(double a,double b)
	{
        return a>b?a:b;
    }

    //粒子群算法
    void PSO(int max)
	{
        for(int i=0;i<max;i++)
		{
            double w=0.4;
            for(int j=0;j<m_n;j++)
			{
                //更新位置和速度，下面就是我们之前重点讲解的两条公式。
				default_random_engine e;
				e.seed(time(0));
				std::normal_distribution<double> Normal_d(0, 1);
				double rd = Normal_d(e);

                m_v[j] = w * m_v[j]+c1 * rd * (m_pbest[j]-m_x[j]) + c2 * rd * (gbest-m_x[j]);
                if(m_v[j]>vmax) m_v[j] = vmax;//控制速度不超过最大值
                m_x[j] += m_v[j];
                
                //越界判断，范围限定在[0, 2]
                if(m_x[j]>2) m_x[j] = 2;
                if(m_x[j]<0) m_x[j] = 0;
                
            }
            fitnessFunction();
            //更新个体极值和群体极值
            for(int j=0;j<m_n;j++)
			{
                m_pbest[j]  =getMAX(m_y[j],m_pbest[j]);
                if(m_pbest[j]>gbest) gbest=m_pbest[j];
                std::cout << "Particle n" << j << ": x = " << m_x[j] <<"  " <<"v = " <<m_v[j] << std::endl;
            }
           std::cout  << (i+1)  << "iter , gbest = "  <<  gbest << std::endl;
        }
        
    }
private:
	int m_n = 2; //粒子个数，这里为了方便演示，我们只取两个，观察其运动方向
	double* m_y;
	double* m_x;
	double* m_v;
	double c1 = 2;
	double c2 = 2;
	double* m_pbest;
	double gbest;
	double vmax = 0.1; //速度最大值
};

int main()
{
	AlgorithmPSO ts;
	ts.init();
	ts.PSO(20);//为了方便演示，我们暂时迭代20次。
	return EXIT_SUCCESS;
}