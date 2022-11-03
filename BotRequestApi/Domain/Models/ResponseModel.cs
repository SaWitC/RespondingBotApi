using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Domain.Models
{
    public class ResponseModel
    {

        [ColumnName("PredictedLabel")]
        public string Message;

        [ColumnName("PredictedResponse")]
        public string Response;

        [ColumnName("Score")]
        public float[] Score;
    }
}
