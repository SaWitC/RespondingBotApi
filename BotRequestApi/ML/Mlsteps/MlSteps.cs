using Domain.Models;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.Mlsteps
{
    public class MlSteps
    {
        public static string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        public static string _trainDataPath = Path.Combine(_appPath, "..", "..","..", "..", "Data","Files", "Requests.tsv");
        public static string _testDataPath = Path.Combine(_appPath, "..", "..","..", "..", "Data","Files", "Test.tsv");
        public static string _modelPath = Path.Combine(_appPath, "..", "..","..", "..", "Data","Files","model.zip");

        public static MLContext _mlContext = new MLContext(seed: 0);
        public static PredictionEngine<RequestModel, ResponseModel> _predEngine;
        public static ITransformer _trainedModel;
        public static IDataView _trainingDataView;
        public static IEstimator<ITransformer> ProccesData()
        {

            // maped to numerical values
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Response", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "MessageFeaturized", inputColumnName: "Message"))
                .Append(_mlContext.Transforms.Concatenate("Features", "MessageFeaturized"))

                .AppendCacheCheckpoint(_mlContext);//for cache using (only for small or midlle datasets)

            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView TrainDataset, IEstimator<ITransformer> pipeline)
        {
            Console.WriteLine("Build and Training");
            var trainingPipeline = pipeline
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));//conver to numerical data

            _trainedModel = trainingPipeline.Fit(TrainDataset);//training
            _predEngine = _mlContext.Model.CreatePredictionEngine<RequestModel, ResponseModel>(_trainedModel);//create model 

            return trainingPipeline;
        }

        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = _mlContext.Data.LoadFromTextFile<RequestModel>(_testDataPath, hasHeader: true);//load data
            // evalute
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));
        }

        public static void SaveModelAsFile(MLContext mLContext, DataViewSchema trainingDataSchema, ITransformer model)
        {
            //save
            mLContext.Model.Save(model, trainingDataSchema, _modelPath);
        }

        //public static void PredictIssue()
        //{
        //    ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

        //    RequestModel singleIssue = new RequestModel() { Message = "Hello" };

        //    _predEngine = _mlContext.Model.CreatePredictionEngine<RequestModel, ResponseModel>(loadedModel);

        //    var prediction = _predEngine.Predict(singleIssue);

        //}

        public static ResponseModel GetResponse(RequestModel testmodel)
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
            _predEngine = _mlContext.Model.CreatePredictionEngine<RequestModel, ResponseModel>(loadedModel);
            var prediction = _predEngine.Predict(testmodel);
            return prediction;
        }
    }
}
