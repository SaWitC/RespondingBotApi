using Domain.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ML.Mlsteps;
using Microsoft.ML;

namespace Services.Services
{
    public class MlService
    {
        public static void UpdateMlContext()
        {     
            MlSteps._trainingDataView = MlSteps._mlContext.Data.LoadFromTextFile<RequestModel>(MlSteps._trainDataPath, hasHeader: true);

            var pipeline = MlSteps.ProccesData();

            var trainingPipelane = MlSteps.BuildAndTrainModel(MlSteps._trainingDataView, pipeline);

            MlSteps.Evaluate(MlSteps._trainingDataView.Schema);

            MlSteps.SaveModelAsFile(MlSteps._mlContext, MlSteps._trainingDataView.Schema, MlSteps._trainedModel);
        }

        public static ResponseModel GetResponse(RequestModel model)
        {
            return MlSteps.GetResponse(model);
        }
    }
}
