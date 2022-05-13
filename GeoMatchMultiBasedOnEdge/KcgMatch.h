/*M///
//
// Author	: KayChan
// Explain	: Shape matching
//
//M*/
 
#ifndef _KCG_MATCH_H_
#define _KCG_MATCH_H_
 
#include <opencv2/opencv.hpp>
#include <omp.h>
 
#ifndef ATTR_ALIGN
#  if defined(__GNUC__)
#    define ATTR_ALIGN(n)	__attribute__((aligned(n)))
#  else
#    define ATTR_ALIGN(n)	__declspec(align(n))
#  endif
#endif // #ifndef ATTR_ALIGN
 
using namespace cv;
using namespace std;
 
namespace kcg_matching {
 
struct MatchRange
{
	float begin;
	float end;
	float step;
 
	MatchRange() : begin(0.f), end(0.f), step(0.f) {}
	MatchRange(float b, float e, float s);
};
inline MatchRange::MatchRange(float b, float e, float s) : begin(b), end(e), step(s) {}
typedef struct MatchRange AngleRange;
typedef struct MatchRange ScaleRange;
 
typedef struct ShapeInfo_S
{
	float angle;
	float scale;
}ShapeInfo;
 
typedef struct Feature_S
{
	int x;
	int y;
	int lbl;
}Feature;
 
typedef struct Candidate_S
{
	/// Sort candidates with high score to the front
	bool operator<(const struct Candidate_S &rhs) const
	{
		return score > rhs.score;
	}
	float score;
	Feature feature;
}Candidate;
 
typedef struct Template_S
{
	int id = 0;
	int pyramid_level = 0;
	int is_valid = 0;
	int x = 0;
	int y = 0;
	int w = 0;
	int h = 0;
	ShapeInfo shape_info;
	vector<Feature> features;
}Template;
 
typedef struct Match_S
{
	/// Sort matches with high similarity to the front
	bool operator<(const struct Match_S &rhs) const
	{
		// Secondarily sort on template_id for the sake of duplicate removal
		if (similarity != rhs.similarity)
			return similarity > rhs.similarity;
		else
			return template_id < rhs.template_id;
	}
 
	bool operator==(const struct Match_S &rhs) const
	{
		return x == rhs.x && y == rhs.y && similarity == rhs.similarity;
	}
 
	int x;
	int y;
	float similarity;
	int template_id;
}Match;
 
typedef enum PyramidLevel_E
{
	PyramidLevel_0 = 0,
	PyramidLevel_1 = 1,
	PyramidLevel_2 = 2,
	PyramidLevel_3 = 3,
	PyramidLevel_4 = 4,
	PyramidLevel_5 = 5,
	PyramidLevel_6 = 6,
	PyramidLevel_7 = 7,
	PyramidLevel_TabooUse = 16,
}PyramidLevel;
 
typedef enum MatchingStrategy_E
{
	Strategy_Accurate = 0,
	Strategy_Middling = 1,
	Strategy_Rough = 2,
}MatchingStrategy;
 
class KcgMatch
{
public:
 
	KcgMatch(string model_root, string class_name);
	~KcgMatch();
	/*
	@model: 输入图像
	@angle_range: 角度范围
	@scale_range: 尺度范围
	@num_features: 特征数
	@weak_thresh：弱阈值
	@strong_thresh: 强阈值
	@mask: 掩码
	*/
	void MakingTemplates(Mat model, AngleRange angle_range, ScaleRange scale_range,
		int num_features, float weak_thresh = 30.0f, float strong_thresh = 60.0f,
		Mat mask = Mat());
	/*
	加载模型
	*/
	void LoadModel();
	/*
	@source: 输入图像
	@score_thresh: 匹配分数阈值
	@overlap: 重叠阈值
	@mag_thresh: 最小梯度阈值
	@greediness: 贪婪度，越小匹配越快，但是可能无法匹配到目标
	@pyrd_level: 金字塔层数，越大匹配越快，但是可能无法匹配到目标
	@T: T参数
	@top_k: 最多匹配多少个
	@strategy: 精确匹配(0), 普通匹配(1), 粗略匹配(2)
	@mask: 匹配掩码
	*/
	vector<Match> Matching(Mat source, float score_thresh = 0.9f, float overlap = 0.4f,
		float mag_thresh = 30.f, float greediness = 0.8f, PyramidLevel pyrd_level = PyramidLevel_3,
		int T = 2, int top_k = 0, MatchingStrategy strategy = Strategy_Accurate, const Mat mask = Mat());
	void DrawMatches(Mat &image, vector<Match> matches, Scalar color);
 
protected:
	void PaddingModelAndMask(Mat &model, Mat &mask, float max_scale);
	vector<ShapeInfo> ProduceShapeInfos(AngleRange angle_range, ScaleRange scale_range);
	Mat Transform(Mat src, float angle, float scale);
	Mat MdlOf(Mat model, ShapeInfo info);
	Mat MskOf(Mat mask, ShapeInfo info);
	void DrawTemplate(Mat &image, Template templ, Scalar color);
	void QuantifyEdge(Mat image, Mat &angle, Mat &quantized_angle, Mat &mag, float mag_thresh, bool calc_180 = true);
	void Quantify8(Mat angle, Mat &quantized_angle, Mat mag, float mag_thresh);
	void Quantify180(Mat angle, Mat &quantized_angle, Mat mag, float mag_thresh);
	Template ExtractTemplate(Mat angle, Mat quantized_angle, Mat mag, ShapeInfo shape_info, 
		PyramidLevel pl, float weak_thresh, float strong_thresh, int num_features, Mat mask);
	Template SelectScatteredFeatures(vector<Candidate> candidates, int num_features, float distance);
	Rect CropTemplate(Template &templ);
	void LoadRegion8Idxes();
	void ClearModel();
	void SaveModel();
	void InitMatchParameter(float score_thresh, float overlap, float mag_thresh, float greediness, int T, int top_k, MatchingStrategy strategy);
	void GetAllPyramidLevelValidSource(Mat &source, PyramidLevel pyrd_level);
	vector<Match> GetTopKMatches(vector<Match> matches);
	vector<Match> DoNmsMatches(vector<Match> matches, PyramidLevel pl, float overlap);
	vector<Match> MatchingPyrd180(Mat src, PyramidLevel pl, vector<int> region_idxes = vector<int>());
	vector<Match> MatchingPyrd8(Mat src, PyramidLevel pl, vector<int> region_idxes = vector<int>());
	void Spread(const Mat quantized_angle, Mat &spread_angle, int T);
	void ComputeResponseMaps(const Mat spread_angle, vector<Mat> &response_maps);
	bool CalcPyUpRoiAndStartPoint(PyramidLevel cur_pl, PyramidLevel obj_pl, Match match,
		Mat &r, Point &p, bool is_padding = false);
	void CalcRegionIndexes(vector<int> &region_idxes, Match match, MatchingStrategy strategy);
	vector<Match> ReconfirmMatches(vector<Match> matches, PyramidLevel pl);
	vector<Match> MatchingFinal(vector<Match> matches, PyramidLevel pl);
 
private:
	typedef vector<Template> TemplateMatchRange;
	TemplateMatchRange templ_all_[PyramidLevel_TabooUse];
	vector<Mat> sources_;
	ATTR_ALIGN(32) float score_table_[180][180];
	ATTR_ALIGN(8) unsigned char score_table_8map_[8][256];
	string model_root_;
	string class_name_;
	AngleRange angle_range_;
	ScaleRange scale_range_;
	vector<int> region8_idxes_;
 
	float score_thresh_;
	float overlap_;
	float mag_thresh_;
	float greediness_;
	int T_;
	int top_k_;
	MatchingStrategy strategy_;
};
}
 
#endif
 